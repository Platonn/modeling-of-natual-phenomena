import cv2
import numpy as np


class MathPendulum:
    def __init__(self, g, mArray, L, ivp):
        self.L = L
        self.g = g
        self.m = mArray
        self.ivp = ivp

        print("mArray.shape", mArray.shape)
        (self.N,) = mArray.shape

        # indexes: # EUCLIDEAN AXIS ORDER IS NORMAL: x,y,z,..
        self.indexX = 0
        self.indexY = 1

    def angular2Euclidean(self, Y):
        euclideanDimensions = 2
        (stepsNum, derivativesNum, ballsNum) = Y.shape

        euclideanY = np.zeros((stepsNum, derivativesNum, ballsNum, euclideanDimensions))
        print("euclideanY.shape", euclideanY.shape)

        theta = Y[:, 0]
        thetaPrim = Y[:, 1]

        euclideanY[:, 0, :, self.indexX] = self.L * np.sin(theta[:])
        euclideanY[:, 0, :, self.indexY] = self.L * np.cos(theta[:])

        euclideanY[:, 1, :, self.indexX] = self.L * thetaPrim[:] * np.sin(theta[:] + np.pi / 2)
        euclideanY[:, 1, :, self.indexY] = self.L * thetaPrim[:] * np.cos(theta[:] + np.pi / 2)

        return euclideanY

    def f(self, t, y):
        (derivativesNum, dimensions) = y.shape
        result = np.zeros(y.shape)
        for i in range(self.N):
            result[0, i] = y[1, i]
            result[1, i] = -self.g / self.L * np.sin(y[0, i])

        return result

    # TODO: TOTALLY REWRITE IT:
    def draw(self, T, euclideanY, box, stepsNum, fps, fileName):
        shape = [800, 800]  # x,y
        video = cv2.VideoWriter(
            fileName,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            tuple(shape))

        def getPos(pos):
            x = pos[self.indexX]
            y = pos[self.indexY]
            resultX = shape[self.indexX] * (x - box['xMin']) / (box['xMax'] - box['xMin'])
            resultY = shape[self.indexY] * (y - box['yMin']) / (box['yMax'] - box['yMin'])
            return np.array([int(resultX), int(resultY)])

        def getVel(vel):
            velX = vel[self.indexX]
            velY = vel[self.indexY]
            resultX = shape[self.indexX] * (velX) / (box['xMax'] - box['xMin'])
            resultY = shape[self.indexY] * (velY) / (box['yMax'] - box['yMin'])
            return np.array([int(resultX), int(resultY)])


        euclideanY[:, 1] *= 0.1  # scale velocity value

        pos_00 = getPos([0, 0])

        for k in range(stepsNum):
            # print("k", k)
            for i in range(self.N):
                # print("i", i)
                # prepare canvas:
                img = np.ones((shape[0], shape[1], 3), np.uint8) * 45
                ball_pos = getPos(euclideanY[k, 0, i])
                ball_vel = getVel(euclideanY[k, 1, i])
                # print("ball_pos", ball_pos)
                # print("ball_velocity", ball_vel)
                # print("ball_pos+ball_velocity", ball_pos + ball_vel)

                lineColor = (240, 240, 240)
                velocityColor = (0, 100, 240)
                circleRadius = 10
                circleColor = (100, 240, 10)

                cv2.line(img,  # position
                         tuple(pos_00),
                         tuple(ball_pos),
                         lineColor, 1)

                cv2.line(img,  # velocity
                         tuple(ball_pos),
                         tuple(ball_pos + ball_vel),
                         velocityColor, 3)

                cv2.circle(img,
                           tuple(ball_pos),
                           circleRadius,
                           circleColor,
                           -1)

                video.write(img.clip(0, 255).astype(np.uint8))
        video.release()
