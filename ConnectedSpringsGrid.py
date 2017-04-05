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

    def powerF(self, A, B):
        d = self.norm(B - A)
        L = self.L
        k = self.k
        result = (
            (B - A) / d * (d - L) * k
        )
        return result

    def powerG(self, velocity):
        alfa = 1  # TODO: values to test: 1,2
        velocity_norm = self.norm(velocity)
        result = - self.r * (velocity_norm ** (alfa - 1)) * velocity
        return result

    def getFunPosBis(self):
        def resultFun(i, j, k, y):
            if i == 0 or i == self.N - 1 or j == 0 or j == self.M - 1:
                return 0  # start and end walls are not accelerating
            else:
                pos_current = y[k - 1, 0, i, j]
                pos_left = y[k - 1, 0, i, j - 1]
                pos_right = y[k - 1, 0, i, j + 1]
                pos_top = y[k - 1, 0, i - 1, j]
                pos_bottom = y[k - 1, 0, i + 1, j]
                posPrim_current = y[k - 1, 1, i, j]

                return (
                           self.powerF(pos_current, pos_left) +
                           self.powerF(pos_current, pos_right) +
                           self.powerF(pos_current, pos_top) +
                           self.powerF(pos_current, pos_bottom) +
                           self.powerG(posPrim_current)

                       ) / self.m[i, j]

        return resultFun

    def eulerExplicit2(self, dimensions, t_start, ivp, t_end, stepsNum):
        h = abs(t_end - t_start) / (stepsNum - 1)
        f = self.getFunPosBis()

        # derivative number, step, posY, posX, value(y/x)=0/1)
        y = np.zeros((stepsNum, dimensions, self.N, self.M, 2))
        t = np.zeros(stepsNum)

        t[0] = t_start
        y[0] = ivp  # y w chwili 0 = ivp

        for k in range(1, stepsNum):  # from 1, because 0 is t_start - handled above
            for i in range(self.N):
                for j in range(self.M):
                    # print('i j', i, j)  # spike
                    t[k] = t_start + k * h

                    # (k-1) means getting value from step before:
                    y[k, 1, i, j] = y[k - 1, 1, i, j] + h * f(i, j, k, y)
                    y[k, 0, i, j] = y[k - 1, 0, i, j] + h * y[k - 1, 1, i, j]  # UWAGA TERAZ JEST PRAWDZIWY EULER
        return t, y

    def draw(self, dimensions, t_start, ivp, t_end, stepsNum, fps, fileName):
        t, y = self.eulerExplicit2(t_start, dimensions, ivp, t_end, stepsNum)

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
            return [int(videoHeight * (y / boxHeight)), int(videoWidth * (x / boxWidth))]

        for k in range(stepsNum):
            # prepare canvas:
            img = np.ones((videoHeight, videoWidth, 3), np.uint8) * 45
            for i in range(self.N):
                for j in range(self.M):
                    spring_current_pos = getPos(y[k, 0, i, j])
                    current_weight = self.m[i, j]
                    # print('k,i,j,x,y', k, i, j, spring_current_pos[self.X], spring_current_pos[self.Y]) #spike

                    spring_left_pos = getPos(y[k, 0, i, j - 1])
                    spring_top_pos = getPos(y[k, 0, i - 1, j])

                    # spike; NIE BEADA SIE WYSWIETLAC LINIE TERAZ!!!!!!!!!!:
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

            # cv2.imshow('step ' + str(k), img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            video.write(img.clip(0, 255).astype(np.uint8))
        video.release()
