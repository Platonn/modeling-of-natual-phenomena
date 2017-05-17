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
    def __init__(self, g, l, m1, k, m2):
        self.g = g
        self.l = l
        self.m1 = m1
        self.k = k
        self.m2 = m2

    def prepareAndGetF(self, ddqs_functions, freedom_coordinants):
        N = len(freedom_coordinants)
        for (_, _, ddq) in freedom_coordinants:
            ddqs_functions[ddq] = ddqs_functions[ddq].subs([
                (Symbol('g'), self.g),
                (Symbol('l'), self.l),
                (Symbol('m1'), self.m1),
                (Symbol('k'), self.k),
                (Symbol('m2'), self.m2),
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
        with open('SlidingPendulum_cachedGetF.py', 'w') as fd:
            fd.write('from numpy import sin, cos, zeros_like\n')
            fd.write('def getF(g, l, m1, k, m2):\n')
            fd.write('    def f(t, y):\n')
            fd.write('        result = zeros_like(y)\n')

            for i in range(N):
                _, _, ddq_i = freedom_coordinants[i]
                fd.write('        result[%d,0] = y[%d,1]\n' % (i, i))
                fd.write('        result[%d,1] = ' % (i,) + str(ddqs_functions[ddq_i]) + '\n')
            fd.write('        return result\n')
            fd.write('    return f\n')

    def getCachedF(self):
        import SlidingPendulum_cachedGetF
        return SlidingPendulum_cachedGetF.getF(self.g, self.l, self.m1, self.k, self.m2)

    @staticmethod
    def mass_to_radius(m, min_r, max_r):
        if (np.max(m) == np.min(m)):
            r = np.ones_like(m) * max_r
        else:
            r = (m - np.min(m)) * ((max_r - min_r) / (np.max(m) - np.min(m))) + min_r
        return r.astype(int)

    @staticmethod
    def render_frame(y, l, m, canvas, meter_scale=1):
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

        def y_to_pos(y, l, pos_start):
            pos = []
            pos.append(pos_start)
            for i in range(len(y)):
                theta = y[i, 0]
                # print(l[i], theta)
                x = pos[i][0] + int(l[i] * np.sin(theta))
                y = pos[i][1] + int(l[i] * np.cos(theta))
                # print(x, y)
                pos.append((x, y))
            return pos

        def _renderCenterPoint(img):
            cv2.line(img, getPos(0, -20), getPos(0, 20), COLOR_CENTER_POINT, 1)
            # cv2.circle(img, getPos(0, 0), 10, COLOR_CENTER_POINT, -1)

        def _renderSpring(img, pos_block):
            cv2.line(img, posCenter, pos_block, COLOR_SPRING, 2)

        def _renderBlock(img, pos_block, m_block):
            scale_block = 1
            m_block_scaled = m_block * scale_block
            cv2.rectangle(
                img,
                (pos_block[0] - m_block_scaled, pos_block[1] - m_block_scaled),
                (pos_block[0] + m_block_scaled, pos_block[1] + m_block_scaled),
                COLOR_BLOCK,
                -1)

        def _renderLine(img, posA, posB):
            cv2.line(img, posA, posB, COLOR_LINE, 2)

        def _renderBall(img, pos, ballId):
            cv2.circle(img, pos, m[ballId], COLOR_BALLS[ballId], -1)

        def _renderBalls():
            for i in range(len(y_b)):
                _renderBall(img, pos[i + 1], i)

        def _renderLines():
            for i in range(len(y_b)):
                _renderLine(img, pos[i], pos[i + 1])


        y_b = y[1:]

        pos_block = getPos(y[0, 0] * meter_scale, 0)
        pos = y_to_pos(y_b, l, pos_block)

        img = canvas.copy()

        _renderCenterPoint(img)
        _renderSpring(img, pos_block)
        _renderBlock(img, pos_block, m[0])
        _renderLines()
        _renderBalls()


        return img

    @staticmethod
    def draw(t, y, l, m, size, fileNameSuffix):
        fps = int(len(t) / t[-1])
        path = 'out/SlidingPendulum_' + fileNameSuffix + '_' + str(time.time()) + '.avi'
        video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (size, size)
        )

        canvas = np.zeros((size, size, 3))

        meter_scale = size / np.sum(l) * 0.3
        l_scaled = l * meter_scale
        m_scaled = SlidingPendulum.mass_to_radius(m, 6, 12)
        for i in range(len(t)):
            frame = SlidingPendulum.render_frame(y[i], l_scaled, m_scaled, canvas, meter_scale)
            video.write(frame.clip(0, 255).astype(np.uint8))
        video.release()

        return path
