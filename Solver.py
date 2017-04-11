import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class Solver:
    @staticmethod
    def solve(method, f, ivp, t_start, t_end, stepsNum):
        h = abs(t_end - t_start) / stepsNum

        shapeY = list(ivp.shape)
        shapeY.insert(0, stepsNum)  # [stepsNum, ...ivpShape]
        Y = np.zeros(tuple(shapeY))

        T = np.arange(t_start, t_end, h)

        Y[0] = ivp  # y in moment 0 = ivp
        for ti in range(1, len(T)):
            Y[ti] = method(f, T[ti - 1], Y[ti - 1], h)
        return T, Y

    @staticmethod
    def euler(f, t, y, h):
        return y + h * f(t, y)

    @staticmethod
    def middlePoint(f, t, y, h):
        K1 = f(t, y)
        K2 = f(t + h * 0.5, y + (h * 0.5 * K1))
        return y + h * K2

    @staticmethod
    def rk4(f, t, y, h):
        K1 = f(t, y)
        K2 = f(t + h * 0.5, y + (h * 0.5 * K1))
        K3 = f(t + h * 0.5, y + (h * 0.5 * K2))
        K4 = f(t + h, y + (h * K3))
        return y + h * (K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)
