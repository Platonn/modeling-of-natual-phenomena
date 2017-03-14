import matplotlib.pyplot as plt
import math

from EulerMethod import *
from ConnectedSprings import *

connectedSprings = ConnectedSprings(np.array([
    # k, L, m
    [None, None, None],
    [1, 1, 1]
]))
# fun_scaffold = lambda t, x, x1, k, m: (-k / m * x)
# fun = lambda t, x, x1: fun_scaffold(t, x, x1, 1, 1)

# ySolution = lambda x: (4 * x + 9 * math.exp(4 * x) - 1) / 8

# eulerMethod = EulerMethod(fun, 0, 0, 20, 100)

t_start = 0
x_start = np.array([1.5])
xPrim_start = np.array([0])
t_end = 20
stepsNum = 100


t, x = connectedSprings.eulerExplicit2(t_start, x_start, xPrim_start, t_end, stepsNum)

plt.plot(t, x, label='connectedSprings.eulerExplicit2')
plt.show()
