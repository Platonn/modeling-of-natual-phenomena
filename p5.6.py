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

###

val_g = 9.80665
val_l = [1]
val_m_block = 1
val_m = [2]
val_k = 5
###


# consts:

ballsNum = len(val_m)

# constants:
k, g = symbols('k, g')
m_block = Symbol("m_block")
m = [Symbol('m[%d]' % (i,)) for i in range(ballsNum)]
l = [Symbol('l[%d]' % (i,)) for i in range(ballsNum)]

# variables:
x, dx, ddx = symbols('x dx ddx')
thetas = [[Symbol('theta[%d]' % (i,)), Symbol('dtheta[%d]' % (i,)), Symbol('ddtheta[%d]' % (i,))] for i in
          range(ballsNum)]

Ek = 0
Ek += 1 / 2 * m_block * (dx ** 2)  # Ek block
current_vel_hori = dx  # TODO: UWAGA, uwzgledniono w v_horizontal dx!
current_vel_vert = 0
for i in range(ballsNum):
    [theta_i, dtheta_i, _] = thetas[i]
    current_vel_vert += l[i] * dtheta_i * sin(theta_i)
    current_vel_hori += l[i] * dtheta_i * cos(theta_i)
    Ek += 0.5 * m[i] * (current_vel_hori ** 2 + current_vel_vert ** 2)

Ep = 0
Ep += 0.5 * k * (x ** 2)  # Ep block
current_h = 0
for i in range(ballsNum):
    [theta_i, _, _] = thetas[i]
    current_h += l[i] * (1 - cos(theta_i))
    Ep += m[i] * g * current_h

L = Ek - Ep

freedom_coordinants = [[x, dx, ddx]] + thetas  # merge lists
print(freedom_coordinants)
#
ddqs_functions = AccelerationEquationsFinder.getFromLagrangian(Ep, Ek, freedom_coordinants)
print(ddqs_functions)

N = len(freedom_coordinants)


slidingPendulum = SlidingPendulum(val_g, val_l, val_m_block, val_k, val_m)
# f = slidingPendulum.prepareAndGetF(ddqs_functions, freedom_coordinants) #SLOW VERSION!!!
slidingPendulum.prepareCachedF(ddqs_functions, freedom_coordinants)
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
# plt.plot(T, Y[:, 1, 0], label='theta[0]')
# plt.legend()
# plt.show()
# plt.plot(T, Y[:, 2, 0], label='theta[1]')
# plt.legend()
# plt.show()

path = SlidingPendulum.draw(T, Y, val_l, val_m, val_m_block, 800, '1')
os.system("xdg-open " + path)
