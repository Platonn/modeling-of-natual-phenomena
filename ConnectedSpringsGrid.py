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

    def powerG(self, xyPrim):
        alfa = 1  # TODO: values to test: 1,2
        xyPrim_norm = self.norm(xyPrim)
        result = - self.r * (xyPrim_norm ** (alfa - 1)) * xyPrim
        return result

    def getFunXYBis(self):
        def resultFun(i, j, xy_current, xy_left, xy_right, xy_top, xy_bottom, xyPrim_current):
            # print('resultFun', j, xj, xj_prev, xj_next) #spike
            if i == 0 or i == self.N - 1 or j == 0 or j == self.M - 1:
                return 0  # start and end walls are not accelerating
            else:
                return (
                           self.powerF(xy_current, xy_left) +
                           self.powerF(xy_current, xy_right) +
                           self.powerF(xy_current, xy_top) +
                           self.powerF(xy_current, xy_bottom) +
                           self.powerG(xyPrim_current)

                       ) / self.m[i, j]

        return resultFun

    def eulerExplicit2(self, t_start, xy_start, xyPrim_start, t_end, stepsNum):
        h = abs(t_end - t_start) / (stepsNum - 1)
        f = self.getFunXYBis()

        # step, posY, posX, value(y/x)=0/1)
        xyBis = np.zeros((stepsNum, self.N, self.M, 2))
        xyPrim = np.zeros((stepsNum, self.N, self.M, 2))
        xy = np.zeros((stepsNum, self.N, self.M, 2))
        t = np.zeros(stepsNum)

        t[0] = t_start
        xyPrim[0, :, :, :] = xyPrim_start[:, :, :]  # w chwili 0 predkosc kazdego ciezarka jest startowa
        xy[0, :, :, :] = xy_start[:, :, :]  # w chwili 0 pozycja kazdego ciezarka jest startowa

        # start walls are not changing position - no matter stepNumber:
        xy[:, 0, :, :] = xy_start[0, :, :]
        xy[:, :, 0, :] = xy_start[:, 0, :]
        # end walls are not changing position - no matter stepNumber:
        xy[:, self.N - 1, :, :] = xy_start[self.N - 1, :, :]
        xy[:, :, self.M - 1, :] = xy_start[:, self.M - 1, :]

        # start and end wall always have zero velocity and acceleration - no matter stepNumber:
        xyBis[:, 0, :] = xyBis[:, self.N - 1, :] = [0, 0]
        xyBis[:, :, 0] = xyBis[:, :, self.M - 1] = [0, 0]
        xyPrim[:, 0, :] = xyPrim[:, self.N - 1, :] = [0, 0]
        xyPrim[:, :, 0] = xyPrim[:, :, self.M - 1] = [0, 0]

        for k in range(1, stepsNum):  # from 1, because 0 is t_start - handled above
            for i in range(1, self.N - 1):
                for j in range(1, self.M - 1):
                    # print('i j', i, j)  # spike
                    t[k] = t_start + k * h

                    # (k-1) means getting value from step before:
                    xy_current = xy[k - 1, i, j]
                    xy_left = xy[k - 1, i, j - 1]
                    xy_right = xy[k - 1, i, j + 1]
                    xy_top = xy[k - 1, i - 1, j]
                    xy_bottom = xy[k - 1, i + 1, j]
                    xyPrim_current = xyPrim[k - 1, i, j]

                    xyBis[k, i, j] = f(
                        # xBis zalezy tylko od pozycji 5 ciezarkow w poprzednim kroku: biezacego i 4 sasiadujacych; oraz od predkosci biezacego ciezarka w poprzenim kroku
                        i, j, xy_current, xy_left, xy_right, xy_top, xy_bottom, xyPrim_current)
                    xyPrim[k, i, j] = xyPrim[k - 1, i, j] + h * xyBis[k, i, j]
                    xy[k, i, j] = xy[k - 1, i, j] + h * xyPrim[k, i, j]
        return t, xy

    def draw(self, t_start, xy_start, xyPrim_start, t_end, stepsNum, fps, fileName):
        t, xy = self.eulerExplicit2(t_start, xy_start, xyPrim_start, t_end, stepsNum)

        videoWidth = 200 * self.M
        videoHeight = 200 * self.N
        video = cv2.VideoWriter(
            fileName,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (videoWidth, videoHeight))

        # cache values:
        rightWall = np.max(xy_start[:, :, self.X])
        leftWall = np.min(xy_start[:, :, self.X])
        bottomWall = np.max(xy_start[:, :, self.Y])
        topWall = np.min(xy_start[:, :, self.Y])

        boxWidth = rightWall - leftWall
        boxHeight = bottomWall - topWall

        def getPos(xy):
            y = xy[self.Y]
            x = xy[self.X]
            return [int(videoHeight * (y / boxHeight)), int(videoWidth * (x / boxWidth))]

        for k in range(stepsNum):
            # prepare canvas:
            img = np.ones((videoHeight, videoWidth, 3), np.uint8) * 45
            for i in range(self.N):
                for j in range(self.M):
                    spring_current_xy = getPos(xy[k, i, j])
                    current_weight = self.m[i, j]
                    # print('k,i,j,x,y', k, i, j, spring_current_xy[self.X], spring_current_xy[self.Y]) #spike

                    spring_left_xy = getPos(xy[k, i, j - 1])
                    spring_top_xy = getPos(xy[k, i - 1, j])

                    # spike; NIE BEADA SIE WYSWIETLAC LINIE TERAZ!!!!!!!!!!:
                    if (i != 0 and j != 0):
                        # springs:
                        lineColor = (240, 240, 240)
                        if (i != self.N):
                            cv2.line(img,
                                     (spring_current_xy[self.X], spring_current_xy[self.Y]),
                                     (spring_left_xy[self.X], spring_left_xy[self.Y]),
                                     lineColor, 1)
                        if (j != self.M):
                            cv2.line(img,
                                     (spring_current_xy[self.X], spring_current_xy[self.Y]),
                                     (spring_top_xy[self.X], spring_top_xy[self.Y]),
                                     lineColor, 1)

                    # elements:
                    circleColor = (240, 240, 240)
                    circleRadius = 6 * (current_weight or 0)
                    cv2.circle(img,
                               (spring_current_xy[self.X], spring_current_xy[self.Y]),
                               circleRadius,
                               circleColor,
                               -1)

            # cv2.imshow('step ' + str(k), img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            video.write(img.clip(0, 255).astype(np.uint8))
        video.release()
