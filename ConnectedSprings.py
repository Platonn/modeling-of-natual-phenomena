import cv2
import numpy as np


class ConnectedSprings:
    def __init__(self, springsParams):
        # 0 index is None
        self.N, _ = springsParams.shape
        self.k = springsParams[:, 0]
        self.L = springsParams[:, 1]
        self.m = springsParams[:, 2]

        print('N', self.N)
        print('k.length', self.k.shape[0])
        print()

    def getFunXBis(self):
        def resultFun(j, xj, xj_prev, xj_next):
            print('resultFun', j, xj, xj_prev, xj_next)
            if j == 0 or j == self.N - 1:
                return 0  # start and end wall are not accelerating
            else:

                return (
                           -self.k[j] * (xj - xj_prev - self.L[j]) +
                           self.k[j + 1] * (xj_next - xj - self.L[j + 1])
                       ) / self.m[j]

        return resultFun

    def eulerExplicit2(self, t_start, x_start, xPrim_start, t_end, stepsNum):
        h = abs(t_end - t_start) / (stepsNum - 1)
        f = self.getFunXBis()

        xBis = np.zeros((stepsNum, self.N))
        xPrim = np.zeros((stepsNum, self.N))
        x = np.zeros((stepsNum, self.N))
        t = np.zeros(stepsNum)

        t[0] = t_start
        xPrim[0, :] = xPrim_start[:]  # w chwili 0 predkosc kazdego ciezarka jest startowa
        x[0, :] = x_start[:]  # w chwili 0 pozycja kazdego ciezarka jest startowa

        # start and end wall are not changing position - no matter stepNumber:
        x[:, 0] = x_start[0]
        x[:, self.N - 1] = x_start[self.N - 1]

        # start and end wall always have zero velocity and acceleration - no matter stepNumber:
        xBis[:, 0] = xBis[:, self.N - 1] = 0
        xPrim[:, 0] = xPrim[:, self.N - 1] = 0

        for k in range(1, stepsNum):  # from 1, because 0 is t_start - handled above
            for j in range(1, self.N - 1):
                print('j', j) #spike
                t[k] = t_start + k * h

                xj = x[k - 1][j]
                xj_prev = x[k - 1][j - 1]
                xj_next = x[k - 1][j + 1]

                xBis[k][j] = f(j, xj, xj_prev, xj_next)  # xBis zalezy tylko od pozycji 3 ciezarkow w poprzednim kroku: biezacego i 2 sasiadujacych
                xPrim[k][j] = xPrim[k - 1][j] + h * xBis[k][j]
                x[k][j] = x[k - 1][j] + h * xPrim[k][j]

        return t, x
