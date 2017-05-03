import cv2
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import os

import time
from sympy import diff, symbols, cos, sin, Derivative, solve, Symbol
from Solver import Solver

# Wyliczyć wzór na theta'' i x'' z langrangiana
# \-> Spróbować najpierw na prostym F = ma ( -kx = m x'' )

# consts:

x, dx, ddx, theta, dtheta, ddtheta = symbols('x dx ddx, theta, dtheta, ddtheta')
k, m1, m2, l, g, t = symbols('k, m1, m2, l, g, t')

Ek_block = 1 / 2 * m1 * (dx ** 2)
Ek_pend = 1 / 2 * m2 * (l ** 2 * dtheta ** 2 + dx ** 2 + 2 * l * dx * dtheta * cos(theta))  ##spike - double check it
Ek = Ek_block + Ek_pend

Ep_block = 1 / 2 * k * (x ** 2)
Ep_pend = - m2 * g * l * cos(theta)
Ep = Ep_block + Ep_pend

L = Ek - Ep

freedom_coordinants = [
    [x, dx, ddx],
    [theta, dtheta, ddtheta]
]
N = len(freedom_coordinants)


def subs_freedom_coordinants_to_being_dependent_on_t(expression, freedom_coordinants):
    for (q, dq, ddq) in freedom_coordinants:
        expression = expression.subs([
            (ddq, Derivative(q(t), t, t)),
            (dq, Derivative(q(t), t)),
            (q, q(t))
        ])
    return expression


def subs_freedom_coordinants_to_being_independent(expression, freedom_coordinants):
    for (q, dq, ddq) in freedom_coordinants:
        expression = expression.subs([
            (Derivative(q(t), t, t), ddq),
            (Derivative(q(t), t), dq),
            (q(t), q)
        ])
    return expression


equations = []
for (q, dq, ddq) in freedom_coordinants:
    diff_L_q = diff(L, q)
    diff_L_dq = diff(L, dq)

    diff_L_dq = subs_freedom_coordinants_to_being_dependent_on_t(diff_L_dq, freedom_coordinants)
    diff_L_dq_t = diff(diff_L_dq, t)
    diff_L_dq_t = subs_freedom_coordinants_to_being_independent(diff_L_dq_t, freedom_coordinants)

    # d/dt dL/dq' = dL/dq
    equationLeftSide = diff_L_dq_t
    equationRightSide = diff_L_q
    equations.append(equationLeftSide - equationRightSide)

ddqs = [freedom_coordinants[i][2] for i in range(N)]
ddqs_functions = solve(equations, ddqs)

# print(equations)
# print(ddqs)
# print(ddqs_functions)

for i in range(N):
    _, _, ddq_i = freedom_coordinants[i]
    # print(i, ddq_i)

    for j in range(N):
        q, dq, ddq = freedom_coordinants[j]

        # subs to freedom_coordinants y
        ddqs_functions[ddq_i] = ddqs_functions[ddq_i].subs([
            (q, Symbol('y[%d,0]' % (j,))),
            (dq, Symbol('y[%d,1]' % (j,)))
        ])
print(ddqs_functions)
# raise Exception()

# write ready function to file:
with open('force_t5.py', 'w') as fd:
    fd.write('from numpy import sin, cos, zeros_like\n')
    fd.write('def force(g, l, m1, k, m2):\n')
    fd.write('    def f(t, y):\n')
    fd.write('        values = zeros_like(y)\n')

    for i in range(N):
        _, _, ddq_i = freedom_coordinants[i]
        fd.write('        values[%d,0] = y[%d,1]\n' % (i, i))
        fd.write('        values[%d,1] = ' % (i,) + str(ddqs_functions[ddq_i]) + '\n')
    fd.write('        return values\n')
    fd.write('    return f\n')

# calculate:
from force_t5 import force

g = 9.80665
l = 1
m1 = 1
m2 = 1
k = 5
f = force(g, l, m1, k, m2)

ivp = np.array([
    [0, 0],
    [1, 0]
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
# path = balls_animation(T, Y, l, M, 800, 0.995, pre='t6')
os.system("xdg-open " + path)
