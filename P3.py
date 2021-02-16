import numpy as np
import sympy

import custom_optimization_library
import matplotlib.pyplot as plt

def main():

    # Define symbolic variables in cost function
    state_sym = {}
    state_sym[0] = sympy.symbols('x1')
    state_sym[1] = sympy.symbols('x2')

    # Define cost function using symbolic variables
    cost_function = []
    cost_function.append(   (state_sym[0]-5)**2 + state_sym[1]**4   )

    constraint_function = []
    constraint_function.append( state_sym[0] + state_sym[1] + 3  )
    constraint_function.append( state_sym[1] - ((state_sym[0]+10)**2 /20) - 2  )

    fig, axs = custom_optimization_library.make_contour_plot(cost_function, [1, 5, 10, 25, 100, 250])
    fig, axs = custom_optimization_library.make_constraint_plot(fig, axs, constraint_function)

    plt.show()



if __name__ == "__main__":
    main()