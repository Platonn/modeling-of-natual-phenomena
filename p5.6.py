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

ballsNum = 1

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
current_vel_hori = dx # TODO: UWAGA, uwzgledniono w v_horizontal dx!
current_vel_vert = 0
for i in range(ballsNum):
    [theta_i, dtheta_i, _] = thetas[i]
    current_vel_vert += l[i] * dtheta_i * sin(theta_i) #spike + np.pi / 2)
    current_vel_hori += l[i] * dtheta_i * cos(theta_i) #spike + np.pi / 2)
    Ek += 0.5 * m[i] * (current_vel_hori ** 2 + current_vel_vert ** 2)  # Ek ball number i

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

# TODO: sprawdzic, czy dobre rownania wychodzą
ddqs_functions = AccelerationEquationsFinder.getFromLagrangian(Ep, Ek, freedom_coordinants)
print(ddqs_functions)

## TODO: jesli wyjda te same rownania, to dostosowac render animacji do wielu kulek, mas, linek, i wartosci y

N = len(freedom_coordinants)

###

val_g = 9.80665
val_l = [1]
val_mBlock = 1
val_mPendulums = [1]
val_k = 5
###

slidingPendulum = SlidingPendulum(val_g, val_l, val_mBlock, val_k, val_mPendulums)
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
print("y.shape", Y.shape)
plt.plot(T, Y[:, 0, 0], label='x')
plt.legend()
plt.show()
plt.plot(T, Y[:, 1, 0], label='theta')
plt.legend()
plt.show()

# M = np.array([m_block, m])
# l = np.array([l, 0])
# path = SlidingPendulum.draw(T, Y, l, M, 800, '1')
# os.system("xdg-open " + path)
