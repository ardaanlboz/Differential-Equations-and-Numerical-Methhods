import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_first_order_diff_eq(diff_eq, t_span, y0, constants={}):

    def eq_wrapper(t, y):
        return diff_eq(t, y, **constants)

    sol = solve_ivp(eq_wrapper, t_span, [y0], dense_output=True)
    
    return sol

def visualizeSolution(sol, t_span, num_points=100, type='exponential'):
    t = np.linspace(t_span[0], t_span[1], num_points)
    y = sol.sol(t)

    plt.plot(t, y.T)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'Solution of {type} Differential Equation')
    plt.grid(True)
    plt.show()

def createDiffEq(params={}, type='exponential'):
    if type == 'exponential':
        return lambda t, y, k: k * y
    elif type == 'logistic':
        return lambda t, y, r, K: r * y * (1 - y / K)
    elif type == 'newtons_cooling':
        return lambda t, T, k, T_env: k * (T_env - T)
    elif type == 'radioactive_decay':
        return lambda t, N, lambda_decay: -lambda_decay * N
    elif type == 'sir_susceptible':
        return lambda t, S, beta, I, N: -beta * S * I / N
    elif type == 'rc_circuit':
        return lambda t, V, R, C, V_in: (V_in - V) / (R * C)
    elif type == 'chemical_reaction':
        return lambda t, A, k: -k * A
    elif type == 'falling_object':
        return lambda t, v, g, c, m: g - (c/m) * v
    elif type == 'population_harvesting':
        return lambda t, P, r, h: r * P - h
    elif type == 'autonomous':
        return lambda t, y: y**2 - y
    else:
        raise ValueError("Invalid type of differential equation")

if __name__ == "__main__":
    # parameters
    t_span = (0, 100)

    # default initial conditions
    initial_conditions = {
        'exponential': {'y0': 0.5},
        'logistic': {'y0': 10},
        'newtons_cooling': {'y0': 90},
        'radioactive_decay': {'y0': 100},
        'sir_susceptible': {'y0': 999},
        'rc_circuit': {'y0': 0}, 
        'chemical_reaction': {'y0': 100}, 
        'falling_object': {'y0': 0},
        'population_harvesting': {'y0': 50},
        'autonomous': {'y0': 2}
    }

    
    list_of_differential_equations = ['exponential', 'logistic', 'newtons_cooling', 'radioactive_decay', 'sir_susceptible', 'rc_circuit', 'chemical_reaction', 'falling_object', 'population_harvesting', 'autonomous']
    
    # default constants
    constants = {
        'exponential': {'k': 0.2},
        'logistic': {'r': 0.5, 'K': 150},
        'newtons_cooling': {'k': -0.07, 'T_env': 25},
        'radioactive_decay': {'lambda_decay': 0.03},
        'sir_susceptible': {'beta': 0.4, 'I': 5, 'N': 1000},
        'rc_circuit': {'R': 50, 'C': 0.05, 'V_in': 10},
        'chemical_reaction': {'k': 0.05},
        'falling_object': {'g': 9.81, 'c': 0.5, 'm': 2},
        'population_harvesting': {'r': 0.02, 'h': 2},
        'autonomous': {}
    }


    for diff_eq in list_of_differential_equations:
        differential = createDiffEq(params=constants[diff_eq], type=diff_eq)
        sol = solve_first_order_diff_eq(differential, t_span, initial_conditions[diff_eq]['y0'], constants[diff_eq])
        visualizeSolution(sol, t_span, num_points=3000, type=diff_eq)

