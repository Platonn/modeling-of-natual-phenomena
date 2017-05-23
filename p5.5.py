import cv2
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import os

import time
from sympy import diff, symbols, cos, sin, Derivative, solve, Symbol

from AccelerationEquationsFinder import *
from SlidingPendulum import *
from Solver import *

# Wyliczyć wzór na theta'' i x'' z langrangiana
# \-> Spróbować najpierw na prostym F = ma ( -kx = m x'' )

# consts:

x, dx, ddx, theta, dtheta, ddtheta = symbols('x dx ddx, theta, dtheta, ddtheta')
k, m1, m2, l, g = symbols('k, m_block, m, l, g')

Ek_block = 0.5 * m1 * (dx ** 2)
Ek_pend = 0.5 * m2 * (l ** 2 * dtheta ** 2 + dx ** 2 + 2 * l * dx * dtheta * cos(theta))  ##spike - double check it
Ek = Ek_block + Ek_pend

Ep_block = 0.5 * k * (x ** 2)
Ep_pend = - m2 * g * l * cos(theta)
Ep = Ep_block + Ep_pend

L = Ek - Ep

freedom_coordinants = [
    [x, dx, ddx],
    [theta, dtheta, ddtheta]
]
print(freedom_coordinants)

ddqs_functions = AccelerationEquationsFinder.getFromLagrangian(Ep, Ek, freedom_coordinants)
print(ddqs_functions)
N = len(freedom_coordinants)


###
g = 9.80665
l = 1
m1 = 1
m2 = 1
k = 5
###

slidingPendulum = SlidingPendulum(g, l, m1, k, m2)
# f = slidingPendulum.prepareAndGetF(ddqs_functions, freedom_coordinants) #SLOW VERSION!!!
# slidingPendulum.prepareCachedF(ddqs_functions, freedom_coordinants)
f = slidingPendulum.getCachedF()

ivp = np.array([
    [0, 0],
    [3, 0]
])

t_start = 0
t_end = 30
stepsNum = 1000

T, Y = Solver.solve(Solver.rk4, f, ivp, t_start, t_end, stepsNum)

# CHARTS:
# print("y.shape", Y.shape)
# plt.plot(T, Y[:, 0, 0], label='x')
# plt.legend()
# plt.show()
# plt.plot(T, Y[:, 1, 0], label='theta')
# plt.legend()
# plt.show()

M = np.array([m1, m2])
l = np.array([l, 0])
path = SlidingPendulum.draw(T, Y, l, M, 800, '1')
os.system("xdg-open " + path)
