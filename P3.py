import numpy as np
import sympy

import custom_optimization_library
import matplotlib.pyplot as plt

def main():

    # Define symbolic variables in cost function
    state_sym = {}
    state_sym[0] = sympy.symbols('x1')
    state_sym[1] = sympy.symbols('x2')

    # Define initial guess
    # -- From the equation we can see there is a minimum at (x1,x2) = (5,0)
    initial_guess = np.array([3., 3.])

    # Define cost function using symbolic variables
    cost_function = []
    cost_function.append(   (state_sym[0]-5)**2 + state_sym[1]**4   )

    constraint_function = []
    constraint_function.append( state_sym[0] + state_sym[1] + 3  )
    constraint_function.append( state_sym[1] - ((state_sym[0]+10)**2 /20) - 2  )

    fig, axs = custom_optimization_library.make_contour_plot(cost_function, [1, 5, 10, 25, 100, 250])
    fig, axs = custom_optimization_library.make_constraint_plot(fig, axs, constraint_function)

    NTCO_state_progression, NTCO_cost_progression, NTCO_time_progression = custom_optimization_library.newton_type_constraint_optimization(cost_function, constraint_function, state_sym, initial_guess)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, NTCO_state_progression, NTCO_cost_progression, NTCO_time_progression, 'GD')
    plt.show()



if __name__ == "__main__":
    main()