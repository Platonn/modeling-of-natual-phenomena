import cv2
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import os

import time
from sympy import diff, symbols, cos, sin, Derivative, solve, Symbol

from AccelerationEquationsFinder import *
from Solver import *


class SlidingPendulum:
    def __init__(self, g, l, m_block, k, m):
        self.g = g
        self.l = l
        self.m_block = m_block
        self.k = k
        self.m = m
        self.N = len(m)

    def prepareAndGetF(self, ddqs_functions, freedom_coordinants):
        N = len(freedom_coordinants)
        for (_, _, ddq) in freedom_coordinants:
            ddqs_functions[ddq] = ddqs_functions[ddq].subs([
                (Symbol('g'), self.g),
                (Symbol('k'), self.k),
                (Symbol('m_block'), self.m_block)
            ])
            for i in range(len(self.m)):
                ddqs_functions[ddq] = ddqs_functions[ddq].subs([
                    (Symbol('m[%d]'%(i,)), self.m[i]),
                    (Symbol('l[%d]'%(i,)), self.l[i])
                ])

        def f(t, y):
            ddqs_functions_numeric = ddqs_functions.copy()
            for i in range(N):
                _, _, ddq = freedom_coordinants[i]

                for j in range(N):
                    ddqs_functions_numeric[ddq] = ddqs_functions_numeric[ddq].subs([
                        (Symbol('y[%d,0]' % (j,)), y[j, 0]),
                        (Symbol('y[%d,1]' % (j,)), y[j, 1])
                    ])
            # print(ddqs_functions)
            # print(ddqs_functions_numeric)

            result = np.zeros_like(y)
            for i in range(N):
                _, _, ddq = freedom_coordinants[i]
                result[i, 0] = y[i, 1]
                result[i, 1] = ddqs_functions_numeric[ddq]
                # print(result[i, 1])
            return result

        return f

    def prepareCachedF(self, ddqs_functions, freedom_coordinants):
        N = len(freedom_coordinants)
        with open('SlidingPendulum_cachedGetF_N%d.py'%(self.N,), 'w') as fd:
            fd.write('from numpy import sin, cos, zeros_like\n')
            fd.write('def getF(g, l, m_block, k, m):\n')
            fd.write('    def f(t, y):\n')
            fd.write('        result = zeros_like(y)\n')

            for i in range(N):
                _, _, ddq_i = freedom_coordinants[i]
                fd.write('        result[%d,0] = y[%d,1]\n' % (i, i))
                fd.write('        result[%d,1] = ' % (i,) + str(ddqs_functions[ddq_i]) + '\n')
            fd.write('        return result\n')
            fd.write('    return f\n')

    def getCachedF(self):
        if self.N==1:
            import SlidingPendulum_cachedGetF_N1 as cached
        elif self.N==2:
            import SlidingPendulum_cachedGetF_N2 as cached
        elif self.N==3:
            import SlidingPendulum_cachedGetF_N3 as cached
        return cached.getF(self.g, self.l, self.m_block, self.k, self.m)

    @staticmethod
    def render_frame(y, l_scaled, m_scaled, m_block_scaled, canvas, scale):
        shape = canvas.shape
        posCenter = (centerX, centerY) = (shape[0] // 2, shape[1] // 2)
        getPos = lambda posX, posY: (centerX + int(posX), centerY + int(posY))
        COLOR_SPRING = [128, 128, 128]
        COLOR_LINE = [64, 64, 64]
        COLOR_BLOCK = [0, 215, 255]
        COLOR_BALLS = [
            [30, 105, 210],
            [0, 64, 255],
            [64, 133, 205]
        ]
        COLOR_CENTER_POINT = COLOR_SPRING
        BALL_RADIUS = 12

        def y_to_pos(y, l, pos_start):
            pos = []
            pos.append(pos_start)
            for i in range(len(y)):
                # print('y:', y)
                theta = y[i, 0]
                pos_x = pos[i][0] + int(l[i] * np.sin(theta))
                pos_y = pos[i][1] + int(l[i] * np.cos(theta))
                # print('x,y:', x, y)
                pos.append((pos_x, pos_y))
            return pos

        def _renderCenterPoint(img):
            cv2.line(img, getPos(0, -20), getPos(0, 20), COLOR_CENTER_POINT, 1)
            # cv2.circle(img, getPos(0, 0), 10, COLOR_CENTER_POINT, -1)

        def _renderSpring(img, pos_block):
            cv2.line(img, posCenter, pos_block, COLOR_SPRING, 2)

        def _renderBlock(img, pos_block, m_block):
            scale_block = 1
            m_block_scaled_half = int(m_block * scale_block / 2)
            cv2.rectangle(
                img,
                (pos_block[0] - m_block_scaled_half, pos_block[1] - m_block_scaled_half),
                (pos_block[0] + m_block_scaled_half, pos_block[1] + m_block_scaled_half),
                COLOR_BLOCK,
                -1)

        def _renderLine(img, posA, posB):
            cv2.line(img, posA, posB, COLOR_LINE, 2)

        def _renderBall(img, pos, ballId):
            cv2.circle(img, pos, int(m_scaled[ballId]), COLOR_BALLS[ballId], -1)

        def _renderBalls():
            for i in range(len(y_b)):
                _renderBall(img, pos[i + 1], i)

        def _renderLines():
            for i in range(len(y_b)):
                _renderLine(img, pos[i], pos[i + 1])


        y_b = y[1:]
        # print('y_b:')
        # print(y_b)

        pos_block = getPos(y[0, 0] * scale, 0)
        pos = y_to_pos(y_b, l_scaled, pos_block)

        img = canvas.copy()

        _renderCenterPoint(img)
        _renderSpring(img, pos_block)
        _renderBlock(img, pos_block, m_block_scaled)
        _renderLines()
        _renderBalls()


        return img

    @staticmethod
    def draw(t, y, l, m, m_block, size, fileNameSuffix):
        fps = int(len(t) / t[-1])
        path = 'out/SlidingPendulum_' + fileNameSuffix + '_' + str(time.time()) + '.avi'
        video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (size, size)
        )

        canvas = np.zeros((size, size, 3))

        scale = size / np.sum(l) * 0.3

        l_scaled = np.array(l) * scale
        m_scaled = np.sqrt(np.array(m) / np.pi) * 20
        m_block_scaled = np.sqrt(m_block) * 20

        for i in range(len(t)):
            # print('y[t_i]:')
            # print(y[i])
            frame = SlidingPendulum.render_frame(y[i], l_scaled, m_scaled,m_block_scaled, canvas, scale)
            video.write(frame.clip(0, 255).astype(np.uint8))
        video.release()

        return path
