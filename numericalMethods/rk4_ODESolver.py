import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return -2*y + 4*t

# ODE solver: 4th order Runge-Kutta method
def rungeKutta4thOrder(f, y0, t0, tf, dt):
    t = np.arange(t0, tf+dt, dt)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = dt * f(t[i-1], y[i-1])
        k2 = dt * f(t[i-1] + dt/2, y[i-1] + k1/2)
        k3 = dt * f(t[i-1] + dt/2, y[i-1] + k2/2)
        k4 = dt * f(t[i] , y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

# Initial conditions
y0 = 1
t0 = 0
tf = 2
dt = 0.1  
# Step size
dt = 0.1

# Solution
t, y = rungeKutta4thOrder(f, y0, t0, tf, dt)

plt.plot(t, y, '-o', label='RK4 Solution')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title("4th Order Runge-Kutta Method Solution of dy/dt = -2*y + 4*t")
plt.legend()
plt.grid(True)
plt.show()
