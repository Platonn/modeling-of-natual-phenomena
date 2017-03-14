import cv2
import numpy as np


class ConnectedSprings:
    def __init__(self, springsParams):
        # 0 index is None
        self.N, _ = springsParams.shape
        self.k = springsParams[:, 0]
        self.L = springsParams[:, 1]
        self.m = springsParams[:, 2]

    def getFunXBis(self):
        def resultFun(j, xj, xj_prev, xj_next):
            if j == 0:
                return 0  # top wall is not moving
            elif j == self.N + 1:
                return np.sum(self.L)  # end wall is not moving - sum of Ls
            else:
                return (
                           -self.k[j] * (xj - xj_prev - self.L[j]) +
                           self.k[j + 1] * (xj_next - xj - self.L[j + 1])
                       ) / self.m[j]

        return resultFun

    def eulerExplicit2(self, t_start, x_start, xPrim_start, t_end, stepsNum):
        h = abs(t_end - t_start) / (stepsNum - 1)
        f = self.getFunXBis()

        x = np.zeros((stepsNum, self.N))
        xBis = np.zeros((stepsNum, self.N))
        xPrim = np.zeros((stepsNum, self.N))
        t = np.zeros(stepsNum)

        t[0] = t_start
        xPrim[0][:] = xPrim_start[:]
        x[0][:] = x_start[:]
        for k in range(1, stepsNum):
            for j in range(1, self.N + 1):
                t[k] = t_start + k * h
                xBis[k][j] = xBis[k - 1][j] + h * f(j, x[k-1][j], x[k-1][j-1], x[k-1][j+1]) #xBis zalezy tylko od pozycji 3 ciezarkow w poprzednim kroku: biezacego i 2 sasiadujacych
                xPrim[k][j] = xPrim[k - 1][j] + h * xBis[k][j]
                x[k][j] = x[k - 1][j] + h * xPrim[k][j]

        return t, x
