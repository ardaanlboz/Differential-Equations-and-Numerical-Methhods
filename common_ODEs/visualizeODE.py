import numpy as np
import matplotlib.pyplot as plt
import os

def visualizeODE(field_function, x_range, y_range, num_points=20, title='Direction Field'):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)

    # Create a grid
    X, Y = np.meshgrid(x_values, y_values)
    
    # Calculate the slope at each point
    U = 1
    V = np.vectorize(field_function)(X, Y)
    
    # Normalize arrows so that they are of equal length
    N = np.sqrt(U**2 + V**2)
    U2, V2 = U/N, V/N

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U2, V2, angles="xy")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()

def simple_field(x, y):
    return x + y

def exponential_field(x, y, k=1.0):
    return k * y

def trig(x, y):
    return np.sin(x) * np.cos(y)

visualizeODE(simple_field, x_range=(-1000, 1000), y_range=(-1000, 1000), num_points=20, title='Direction Field')
visualizeODE(lambda x, y: exponential_field(x, y, k=1.0), x_range=(-1000, 1000), y_range=(-1000, 1000), num_points=20, title='Direction Field')
visualizeODE(trig, x_range=(-1000, 1000), y_range=(-1000, 1000), num_points=20, title='Direction Field')
