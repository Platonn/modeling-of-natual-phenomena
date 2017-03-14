import matplotlib.pyplot as plt
import math

from EulerMethod import *

fun_scaffold = lambda t, x, x1, k, m: (-k / m * x)
fun = lambda t, x, x1: fun_scaffold(t, x, x1, 1, 1)

ySolution = lambda x: (4 * x + 9 * math.exp(4 * x) - 1) / 8

eulerMethod = EulerMethod(fun, 0, 0, 20, 100)

t, y1 = eulerMethod.explicit2(y1_start=2)  # initial velocity
plt.plot(t, y1, label='explicit')
plt.show()
