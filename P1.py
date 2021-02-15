import numpy as np
import sympy

import custom_optimization_library

def main():

    # Define symbolic variables in cost function
    state_sym = {}
    state_sym[0] = sympy.symbols('x1')
    state_sym[1] = sympy.symbols('x2')

    # Define cost function using symbolic variables
    cost_function = []
    cost_function.append(   (state_sym[0]-5)**2 + state_sym[1]**4   )

    fig, axs = custom_optimization_library.make_contour_plot(cost_function, [1, 5, 10, 25, 100, 250])

    # Define initial guess
    # -- From the equation we can see there is a minimum at (x1,x2) = (5,0)
    initial_guess = np.array([3., 3.])

    # Gradient Descent w/ Fixed Gamma
    GD_state_progression, GD_cost_progression, GD_time_progression = custom_optimization_library.gradient_descent(cost_function, state_sym, initial_guess)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, GD_state_progression, GD_cost_progression, GD_time_progression, 'GD')

    # Gradient descent with line-search globalization

    # Gradient descent using Barzilai-Borwein Equation

    # True Hessian

    # Gauss-Newton

    # Levenberg-Marquardt

    # Constant Hessian

    # BFGS



if __name__ == "__main__":
    main()







