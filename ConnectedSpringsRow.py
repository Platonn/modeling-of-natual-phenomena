import cv2
import numpy as np
import matplotlib.pyplot as plt



class ConnectedSpringsRow:
    def __init__(self, springsParams):
        # 0 index is None
        self.N, _ = springsParams.shape
        self.k = springsParams[:, 0]
        self.L = springsParams[:, 1]
        self.m = springsParams[:, 2]

        # spike:
        # print('N', self.N)
        # print('k.length', self.k.shape[0])
        # print()

    def getFunXBis(self):
        def resultFun(j, xj, xj_prev, xj_next):
            # print('resultFun', j, xj, xj_prev, xj_next) #spike
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
                # print('j', j)  # spike
                t[k] = t_start + k * h

                xj = x[k - 1][j]
                xj_prev = x[k - 1][j - 1]
                xj_next = x[k - 1][j + 1]

                xBis[k][j] = f(j, xj, xj_prev,
                               xj_next)  # xBis zalezy tylko od pozycji 3 ciezarkow w poprzednim kroku: biezacego i 2 sasiadujacych
                xPrim[k][j] = xPrim[k - 1][j] + h * xBis[k][j]
                x[k][j] = x[k - 1][j] + h * xPrim[k][j]

        return t, x

    def draw(self, t_start, x_start, xPrim_start, t_end, stepsNum, fps):
        t, x = self.eulerExplicit2(t_start, x_start, xPrim_start, t_end, stepsNum)

        videoWidth = 640
        videoHeight = 30


        video = cv2.VideoWriter(
            'out/'
            'connectedSprings.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps, (videoWidth, videoHeight))

        for k in range(stepsNum):
            # prepare cavas:
            img = np.ones((videoHeight, videoWidth, 3), np.uint8) * 255
            pos_y = int(videoHeight / 2)


            # springs free-state positions:
            for j in range(1, self.N-1):
                spring_free_pos_x = int(videoWidth * (np.sum(self.L[1:j]) / (np.max(x_start) - np.min(x_start))))
                cv2.circle(img, (spring_free_pos_x, pos_y), 3+self.m[j], (0, 255, 0), -1)


            for j in range(self.N):
                if (j != 0):
                    # spring:
                    spring_left_x = int(videoWidth * (x[k, j - 1] / (np.max(x_start) - np.min(x_start))))
                    spring_right_x = int(videoWidth * (x[k, j] / (np.max(x_start) - np.min(x_start))))

                    max_k = np.max(self.k[1:])
                    k_factor = (max_k - self.k[j]+1)
                    color = (0, 30+k_factor*50, 255)

                    cv2.line(img, (spring_left_x, pos_y), (spring_right_x, pos_y), color, 3)

                # elements:
                pos_x = int(videoWidth * (x[k, j] / (np.max(x_start) - np.min(x_start))))
                cv2.circle(img, (pos_x, pos_y), 6, (255, 0, 0), -1)


            # cv2.imshow('step ' + str(k), img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            video.write(img.clip(0, 255).astype(np.uint8))
        video.release()


        #plot:
        for j in range(0, self.N):
            # print('t.shape', t.shape)
            # print('x[:][j].shape', x[:, j].shape)
            # print('------------')
            plt.plot(t, x[:, j], label='x' + str(j))

        # plt.plot(t, t, label='t') #spike
        plt.legend()
        plt.show()
