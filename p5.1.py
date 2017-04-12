import matplotlib.pyplot as plt
import math

from Solver import *
from MathPendulum import *

# TODO:
# 1. Animation
# 2. Plot thetaPrim = f(theta)

L = 1  # spike: np.array([1])
dim = 2  # dimensions
derivativesNum = 2
m = np.array([1])
g = 10  # spike 9.81
r = 0.1

# prepare ivp:
y0 = np.array([np.pi / 2])
y1 = np.array([0])
ivp = np.array([y0, y1])

mathPendulum = MathPendulum(g, m, L, r, ivp)

t_start = 0

t_end = 20
stepsNum = 1000

fps = stepsNum / 20

T, Y = Solver.solve(Solver.rk4, mathPendulum.f, mathPendulum.ivp, t_start, t_end, stepsNum)

euclideanY = mathPendulum.angular2Euclidean(Y)

# # debug theta
# # print("y.shape", Y.shape)
# # plt.plot(T, Y[:, 0], label='theta')
# # plt.legend()
# # plt.show()
# # # debug thetaPrim
# # plt.plot(T, Y[:, 1], label='thetaPrim')
# # plt.legend()
# # plt.show()

# debug theta to thetaPrim
# plt.plot(Y[:, 0], Y[:, 1], label='theta to thetaPrim')
# plt.legend()
# plt.show()
#
# debug euclidean X
# print("y.shape", euclideanY.shape)
# plt.plot(T, euclideanY[:, 0, 0, mathPendulum.indexX], label='euclidean x')
# plt.legend()
# plt.show()
# # debug euclidean Y
# plt.plot(T, euclideanY[:, 0, 0, mathPendulum.indexY], label='euclidean y')
# plt.legend()
# plt.show()

# draw:
fileName = 'out/MathPendulum.avi'

boxModifier = 1.1
box = {
    'xMin': -L * 1.5,
    'xMax': L * 1.5,
    'yMin': -L * 1.5,
    'yMax': L * 1.5
}

mathPendulum.draw(T, euclideanY, box, stepsNum, fps, fileName)

os.system("xdg-open " + fileName)
