import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return -2*y + 4*t

# ODE solver: Implicit Euler method
def implicit_euler_step(y, h, f, t_next):
    # Not adaptive, guessing the next value of y
    y_next = y
    for _ in range(N):
        y_next = y + h * f(t_next, y_next)
    return y_next

# Initial conditions
y0 = 1
t0 = 0
tf = 5
# Step size
N = 50 # Number of steps
h = (tf - t0) / N

# Time and solution arrays
times = np.linspace(t0, tf, N+1)
y = np.zeros(N+1)
y[0] = y0

# Time-stepping loop
for n in range(N):
    t_next = times[n+1]
    y[n+1] = implicit_euler_step(y[n], h, f, t_next)

# Plotting the solution
plt.plot(times, y, label="Implicit Euler Solution")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title(f"Implicit Euler Method Solution of dy/dt = -2*y + 4*t, stepsize {h}")
plt.legend()
plt.grid(True)
plt.show()
