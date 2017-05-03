from numpy import sin, cos, zeros_like
def force(g, l, m1, k, m2):
    def f(t, y):
        values = zeros_like(y)
        values[0,0] = y[0,1]
        values[0,1] = (0.5*g*m2*sin(2.0*y[1,0]) - k*y[0,0] + l*m2*y[1,1]**2*sin(y[1,0]))/(m1 + m2*sin(y[1,0])**2)
        values[1,0] = y[1,1]
        values[1,1] = -(g*(m1 + m2)*sin(y[1,0]) + (-k*y[0,0] + l*m2*y[1,1]**2*sin(y[1,0]))*cos(y[1,0]))/(l*(m1 + m2*sin(y[1,0])**2))
        return values
    return f
