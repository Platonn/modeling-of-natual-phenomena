import matplotlib.pyplot as plt
import math

from ConnectedSpringsGrid import *
from EulerMethod import *

N = 5
M = 5
L = 1
dim = 2  # dimensions
r = 0.001
k = 4

m = np.array([[1 for i in range(M)] for j in range(N)], np.float64)

y0 = np.array([[[L * j, L * i] for i in range(M)] for j in range(N)], np.float64)
y0[1, 1] += [1, 1]

y1 = np.zeros((N, M, dim))

ivp = np.array([y0, y1])

connectedSpringsGrid = ConnectedSpringsGrid(k=k, L=L, r=r, mArray=m)

xy_start = ivp[0]
xyPrim_start = ivp[1]

t_start = 0

t_end = 20
stepsNum = 1000

fps = stepsNum / 20

# t, xy = connectedSpringsGrid.eulerExplicit2(t_start, dim, ivp, t_end, stepsNum)
# # debug X:
# for i in range(0, connectedSpringsGrid.N):
#     for j in range(0, connectedSpringsGrid.M):
#         plt.plot(t, xy[:, i, j, 1], label='x' + str(i) + str(j))
# plt.legend()
# plt.show()

# # debug Y:
# for i in range(0, connectedSpringsGrid.N):
#     for j in range(0, connectedSpringsGrid.M):
#         plt.plot(t, xy[:, i, j, 0], label='y' + str(i) + str(j))
# plt.legend()
# plt.show()


# draw:
fileName = 'out/connectedSpringsGrid.avi'
connectedSpringsGrid.draw(t_start, dim, ivp, t_end, stepsNum, fps, fileName)

os.system("xdg-open " + fileName)
