import cv2
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import os

import time
from sympy import diff, symbols, cos, sin, Derivative, solve, Symbol
from Solver import Solver

class AccelerationEquationsFinder:
    @staticmethod
    def getFromLagrangian(Ep, Ek, freedom_coordinants):
        L = Ek - Ep
        N = len(freedom_coordinants)

        equations = []
        for (q, dq, ddq) in freedom_coordinants:
            diff_L_q = diff(L, q)
            diff_L_dq = diff(L, dq)

            diff_L_dq = AccelerationEquationsFinder._subs_freedom_coordinants_to_being_dependent_on_t(diff_L_dq, Symbol('t'), freedom_coordinants)
            diff_L_dq_t = diff(diff_L_dq, Symbol('t'))
            diff_L_dq_t = AccelerationEquationsFinder._subs_freedom_coordinants_to_being_independent(diff_L_dq_t, Symbol('t'), freedom_coordinants)

            # d/dt dL/dq' = dL/dq
            equationLeftSide = diff_L_dq_t
            equationRightSide = diff_L_q
            equations.append(equationLeftSide - equationRightSide)

        ddqs = [freedom_coordinants[i][2] for i in range(N)]
        ddqs_functions = solve(equations, ddqs)

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
        # print(ddqs_functions)
        return ddqs_functions

    @staticmethod
    def _subs_freedom_coordinants_to_being_dependent_on_t(expression, t, freedom_coordinants):
        for (q, dq, ddq) in freedom_coordinants:
            expression = expression.subs([
                (ddq, Derivative(q(t), t, t)),
                (dq, Derivative(q(t), t)),
                (q, q(t))
            ])
        return expression

    @staticmethod
    def _subs_freedom_coordinants_to_being_independent(expression, t, freedom_coordinants):
        for (q, dq, ddq) in freedom_coordinants:
            expression = expression.subs([
                (Derivative(q(t), t, t), ddq),
                (Derivative(q(t), t), dq),
                (q(t), q)
            ])
        return expression
