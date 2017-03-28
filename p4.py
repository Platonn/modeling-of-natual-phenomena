import matplotlib.pyplot as plt
import math

from ConnectedSpringsGrid import *
from EulerMethod import *

connectedSpringsGrid = ConnectedSpringsGrid(
    k=4,
    L=1,
    r=0.001,
    mArray=np.array([
        [None, None, None, None],
        [None, 1, 2, None],
        [None, None, None, None]
    ])
)

# zagadnienie poczatkowe:
# pozycja ma 2 wspolrzedne:
xy_start = np.array([
    [[0, 0], [0, 1], [0, 2], [0, 3]],
    [[1, 0], [1.5, 1.5], [1.2, 2], [1, 3]],
    [[2, 0], [2, 1], [2, 2], [2, 3]]
])
# predkosc ma 2 wspolrzedne:
xyPrim_start = np.array([
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 0], [0, 0]]
])
t_start = 0

t_end = 20
stepsNum = 1000

fps = stepsNum / 20

t, xy = connectedSpringsGrid.eulerExplicit2(t_start, xy_start, xyPrim_start, t_end, stepsNum)

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
connectedSpringsGrid.draw(t_start, xy_start, xyPrim_start, t_end, stepsNum, fps, fileName)

os.system("xdg-open " + fileName)
