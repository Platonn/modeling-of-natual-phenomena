import cv2
import numpy as np

from MathPendulum import MathPendulum

class MathPendulumSimplified(MathPendulum):
    def f(self, t, y):
        (derivativesNum, dimensions) = y.shape
        result = np.zeros(y.shape)
        for i in range(self.N):
            result[0, i] = y[1, i]

            gravityPendulumForce = -self.g / self.L * y[0, i] #simplified = not np.sin(y[0,i]
            resistanceForce = - self.resist * self.L * y[1, i]
            result[1, i] = gravityPendulumForce + resistanceForce

        return result

