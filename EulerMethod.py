import numpy as np
import sympy as sp


# Euler method is approximating solution y(t) in interval [t0, b] in equal distances
class EulerMethod:
    def __init__(self, f, t_start, y_start, t_end, N):
        self.f = f
        self.t0 = t_start
        self.y0 = y_start
        self.b = t_end
        self.N = N
        self.h = abs(t_end - t_start) / (N - 1)

    def explicit(self):
        y = np.zeros(self.N)
        t = np.zeros(self.N)

        y[0] = self.y0
        t[0] = self.t0
        for k in range(1, self.N):
            t[k] = self.t0 + k * self.h
            y[k] = y[k - 1] + self.h * self.f(t[k - 1], y[k - 1])

        return t, y

    def hibridTrapeze(self):
        tExplicit, yExplicit = self.explicit()

        y = np.zeros(self.N)
        t = np.zeros(self.N)

        y[0] = self.y0
        t[0] = self.t0
        for k in range(1, self.N):
            t[k] = self.t0 + k * self.h
            y[k] = y[k - 1] + \
                   self.h / 2 * self.f(t[k - 1], y[k - 1]) + \
                   self.h / 2 * self.f(tExplicit[k], yExplicit[k])

        return t, y

    def backwarded(self, equation, symbols):
        yp_sym, y_sym, t_sym = symbols

        f = sp.solve(equation, yp_sym)[0] # na wszelki wypadek rozwiklaj

        y = np.zeros(self.N)
        t = np.zeros(self.N)

        y[0] = self.y0
        t[0] = self.t0
        for k in range(1, self.N):
            t[k] = self.t0 + k * self.h
            y_k = sp.symbols('y_k')
            eq = sp.Eq(y_k, y[k - 1] + self.h * f.subs([(t_sym, t[k]), (y_sym, y_k)]))
            y[k] = sp.solve(eq, y_k)[0]

        return t, y

    def calcFromSolution(self, ySolution):
        y = np.zeros(self.N)
        t = np.zeros(self.N)

        y[0] = self.y0
        t[0] = self.t0
        for k in range(1, self.N):
            t[k] = self.t0 + k * self.h
            y[k] = ySolution(t[k])

        return t, y

    def explicit2(self, y1_start):
        self.y1_start = y1_start

        y = np.zeros(self.N)
        y1 = np.zeros(self.N)
        t = np.zeros(self.N)

        t[0] = self.t0
        y1[0] = self.y1_start
        y[0] = self.y0
        for k in range(1, self.N):
            t[k] = self.t0 + k * self.h
            y1[k] = y1[k - 1] + self.h * self.f(t[k - 1], y[k - 1], y1[k - 1])
            y[k] = y[k - 1] + self.h * y1[k]

        return t, y
