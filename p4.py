import matplotlib.pyplot as plt
import math

from Solver import *
from ConnectedSpringsGrid import *

N = 3
M = 3
L = 1
dim = 2  # dimensions
r = 0.001
k = 4
derivativesNum = 2

m = np.array([[1 for i in range(M)] for j in range(N)], np.float64)

# prepare ivp:
y0 = np.array([[[L * j, L * i] for i in range(M)] for j in range(N)], np.float64)
y0[1, 1] += [0.5, 0.5]

y1 = np.zeros((N, M, dim))

ivp = np.array([y0, y1])

connectedSpringsGrid = ConnectedSpringsGrid(k=k, L=L, r=r, mArray=m, ivp=ivp)

t_start = 0

t_end = 20
stepsNum = 1000

fps = stepsNum / 20

T, Y = Solver.solve(Solver.rk4, connectedSpringsGrid.f, connectedSpringsGrid.ivp, t_start, t_end, stepsNum)


# debug X without walls:
print("t.shape", T.shape)
print("y.shape", Y.shape)
for i in range(1, connectedSpringsGrid.N-1):
    for j in range(1, connectedSpringsGrid.M-1):
        plt.plot(T, Y[:, 0, i, j, 1], label='x' + str(i) + str(j))
plt.legend()
plt.show()
# debug Y without walls:
for i in range(1, connectedSpringsGrid.N-1):
    for j in range(1, connectedSpringsGrid.M-1):
        plt.plot(T, Y[:, 0, i, j, 0], label='y' + str(i) + str(j))
plt.legend()
plt.show()

# draw:
fileName = 'out/connectedSpringsGrid.avi'
connectedSpringsGrid.draw(T, Y, stepsNum, fps, fileName)


os.system("xdg-open " + fileName)
