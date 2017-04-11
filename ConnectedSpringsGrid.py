import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ConnectedSpringsGrid:
    def __init__(self, k, L, r, mArray):
        self.L = L
        self.k = k
        self.r = r

        # 0 index is None and last is None
        (self.N, self.M) = mArray.shape
        self.m = mArray

        # indexes:
        self.Y = 0
        self.X = 1

    def norm(self, A):
        return np.linalg.norm(A)

    def sumPowersF(self, i, j, y):
        pos_current = y[0][i, j]
        pos_left = y[0][i, j - 1]
        pos_right = y[0][i, j + 1]
        pos_top = y[0][i - 1, j]
        pos_bottom = y[0][i + 1, j]

        return (
            self.powerF(pos_current, pos_left) +
            self.powerF(pos_current, pos_right) +
            self.powerF(pos_current, pos_top) +
            self.powerF(pos_current, pos_bottom)
        )

    def powerF(self, A, B):
        d = self.norm(B - A)
        L = self.L
        k = self.k
        result = (
            (B - A) / d * (d - L) * k
        )
        return result

    def powerG(self, i, j, y):
        velocity = y[1][i, j]
        alfa = 1  # TODO: values to test: 1,2
        velocity_norm = self.norm(velocity)
        result = - self.r * (velocity_norm ** (alfa - 1)) * velocity
        return result

    def f(self, t, y):
        (derivativesNum, N, M, dimensions) = y.shape
        result = np.zeros(y.shape)
        for i in range(N):
            for j in range(M):
                result[0, i, j] = y[1, i, j]

                if i == 0 or i == self.N - 1 or j == 0 or j == self.M - 1:
                    result[1, i, j] = 0  # start and end walls have zero velocity
                else:
                    result[1, i, j] = (self.sumPowersF(i, j, y) + self.powerG(i, j, y)) / self.m[i, j]

        return result

    def eulerExplicit2(self, derivativesNum, t_start, ivp, t_end, stepsNum):
        h = abs(t_end - t_start) / (stepsNum)
        Y = np.zeros((stepsNum, derivativesNum, self.N, self.M, 2))
        T = np.arange(t_start, t_end, h)

        Y[0] = ivp  # y in moment 0 = ivp
        for ti in range(1, len(T)):
            Y[ti] = self.naiveEulerMethod(self.f, T[ti - 1], Y[ti - 1], h)
        return T, Y

    def naiveEulerMethod(self, f, t, y, h):
        return y + h * f(t, y)

    def draw(self, derivativesNum, t_start, ivp, t_end, stepsNum, fps, fileName):
        t, y = self.eulerExplicit2(derivativesNum, t_start, ivp, t_end, stepsNum)

        videoWidth = 200 * self.M
        videoHeight = 200 * self.N
        video = cv2.VideoWriter(
            fileName,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (videoWidth, videoHeight))

        # cache values:
        y0 = ivp[0]
        rightWall = np.max(y0[:, :, self.X])
        leftWall = np.min(y0[:, :, self.X])
        bottomWall = np.max(y0[:, :, self.Y])
        topWall = np.min(y0[:, :, self.Y])

        boxWidth = rightWall - leftWall
        boxHeight = bottomWall - topWall

        def getPos(pos):
            y = pos[self.Y]
            x = pos[self.X]
            resultY = videoHeight * (y / boxHeight)
            resultX = videoWidth * (x / boxWidth)
            return [int(resultY), int(resultX)]

        for k in range(stepsNum):
            # prepare canvas:
            img = np.ones((videoHeight, videoWidth, 3), np.uint8) * 45
            for i in range(self.N):
                for j in range(self.M):
                    spring_current_pos = getPos(y[k, 0, i, j])
                    current_weight = self.m[i, j]

                    spring_left_pos = getPos(y[k, 0, i, j - 1])
                    spring_top_pos = getPos(y[k, 0, i - 1, j])

                    if (i != 0 and j != 0):
                        # springs:
                        lineColor = (240, 240, 240)
                        if (i != self.N):
                            cv2.line(img,
                                     (spring_current_pos[self.X], spring_current_pos[self.Y]),
                                     (spring_left_pos[self.X], spring_left_pos[self.Y]),
                                     lineColor, 1)
                        if (j != self.M):
                            cv2.line(img,
                                     (spring_current_pos[self.X], spring_current_pos[self.Y]),
                                     (spring_top_pos[self.X], spring_top_pos[self.Y]),
                                     lineColor, 1)

                    # elements:
                    circleColor = (240, 240, 240)
                    circleRadius = 6 * (int(current_weight) or 0)
                    cv2.circle(img,
                               (spring_current_pos[self.X], spring_current_pos[self.Y]),
                               circleRadius,
                               circleColor,
                               -1)

            video.write(img.clip(0, 255).astype(np.uint8))
        video.release()
