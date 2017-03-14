import matplotlib.pyplot as plt
import math

from EulerMethod import *
from ConnectedSprings import *

connectedSprings = ConnectedSprings(np.array([
    # k, L, m
    # numeracja sprezyn bedzie potem od 1 do N
    [None, None, None],
    [1, 1, 1],
    [1, 1, None]  # ostatnia sprezyna istnieje(!) - ma swoje k i L
]))

# zagadnienie poczatkowe:
x_start = np.array([0, 1.5, 2])
xPrim_start = np.array([0, 0, 0])
t_start = 0

t_end = 20
stepsNum = 1000

t, x = connectedSprings.eulerExplicit2(t_start, x_start, xPrim_start, t_end, stepsNum)

for j in range(0, connectedSprings.N):
    # print('t.shape', t.shape)
    # print('x[:][j].shape', x[:, j].shape)
    # print('------------')
    plt.plot(t, x[:, j], label='x' + str(j))

# plt.plot(t, t, label='t') #spike
plt.legend()
plt.show()
