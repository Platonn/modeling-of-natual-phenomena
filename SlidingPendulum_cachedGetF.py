from numpy import sin, cos, zeros_like
def getF(g, l, m_block, k, m):
    def f(t, y):
        result = zeros_like(y)
        result[0,0] = y[0,1]
        result[0,1] = (0.5*g*m[0]*sin(2.0*y[1,0]) - k*y[0,0] + l[0]*m[0]*y[1,1]**2*sin(y[1,0]))/(m[0]*sin(y[1,0])**2 + m_block)
        result[1,0] = y[1,1]
        result[1,1] = -(g*(m[0] + m_block)*sin(y[1,0]) + (-k*y[0,0] + l[0]*m[0]*y[1,1]**2*sin(y[1,0]))*cos(y[1,0]))/(l[0]*(m[0]*sin(y[1,0])**2 + m_block))
        return result
    return f
