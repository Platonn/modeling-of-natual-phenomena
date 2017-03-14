import matplotlib.pyplot as plt
import sympy as sp
import math

from EulerMethod import *

fun = lambda t, y: (1 - 2 * t + 4 * y)
ySolution = lambda x: (4 * x + 9 * math.exp(4 * x) - 1) / 8

eulerMethod = EulerMethod(fun, 0, 1, 2, 100)
tExplicit, yExplicit = eulerMethod.explicit()
tHibridTrapeze, yHibridTrapeze = eulerMethod.hibridTrapeze()

symbols = sp.symbols('yp y t')
yp, y, t = symbols
equation = sp.Eq(yp, 1 - 2*t + 4*y)
tBackwarded, yBackwarded = eulerMethod.backwarded(equation, symbols)

tOrigin, yOrigin = eulerMethod.calcFromSolution(ySolution)

plt.plot(tExplicit, yExplicit, label='explicit')
plt.plot(tHibridTrapeze, yHibridTrapeze, label='hibrid trapeze')
plt.plot(tBackwarded, yBackwarded, label='backwarded')
plt.plot(tOrigin, yOrigin, label='origin')
plt.legend(loc=2)

plt.show()
