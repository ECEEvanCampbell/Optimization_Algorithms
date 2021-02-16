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
    cost_function.append(   sympy.sin(state_sym[0]) + sympy.cos(state_sym[1]) + 3.14159   )

    fig, axs = custom_optimization_library.make_contour_plot(cost_function, np.arange(1.,5.,0.5))

    # Define initial guess
    # -- From the equation we can see there is a minimum at (x1,x2) = (5,0)
    initial_guess = np.array([3., 3.])

    # Gradient Descent w/ Fixed Gamma
    GD_state_progression, GD_cost_progression, GD_time_progression = custom_optimization_library.gradient_descent(cost_function, state_sym, initial_guess)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, GD_state_progression, GD_cost_progression, GD_time_progression, 'GD')

    # Gradient descent with line-search globalization
    GDLS_state_progression, GDLS_cost_progression, GDLS_time_progression = custom_optimization_library.gradient_descent_line_search(cost_function, state_sym, initial_guess, np.logspace(-5,0,10))
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, GDLS_state_progression, GDLS_cost_progression, GDLS_time_progression, 'GDLS')

    # TODO: address how to do this
    # Gradient descent using Barzilai-Borwein Equation
    #GDBB_state_progression, GDBB_cost_progression, GDBB_time_progression = custom_optimization_library.gradient_descent_barzilai_borwein(cost_function, state_sym, initial_guess)
    #fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, GDBB_state_progression, GDBB_cost_progression, GDBB_time_progression, 'GDBB')

    # True Hessian
    NTO_state_progression, NTO_cost_progression, NTO_time_progression = custom_optimization_library.newton_type_optimization(cost_function, state_sym, initial_guess, enforce_PD=True)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, NTO_state_progression, NTO_cost_progression, NTO_time_progression, 'NTO')
    
    # Gauss-Newton -- Cannot specify an r array
    #r = UNKNOWN
    #GN_state_progression, GN_cost_progression, GN_time_progression = custom_optimization_library.gauss_newton_optimization(cost_function, r, state_sym, initial_guess)
    #fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, GN_state_progression, GN_cost_progression, GN_time_progression, 'GN')

    # Levenberg-Marquardt
    #r = UNKNOWN
    #LM_state_progression, LM_cost_progression, LM_time_progression = custom_optimization_library.levenberg_marquardt_optimization(cost_function, r, state_sym, initial_guess)
    #fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, LM_state_progression, LM_cost_progression, LM_time_progression, 'LM')

    # Constant Hessian
    hessian_val = np.array([[3e-1, 0], [0, 3e-2]])
    CH_state_progression, CH_cost_progression, CH_time_progression = custom_optimization_library.constant_hessian_optimzation(cost_function, hessian_val, state_sym, initial_guess)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, CH_state_progression, CH_cost_progression, CH_time_progression, 'CH')

    # BFGS
    BFGS_state_progression, BFGS_cost_progression, BFGS_time_progression = custom_optimization_library.BFGS_optimization(cost_function, state_sym, initial_guess, enforce_PD=True)
    fig, axs = custom_optimization_library.plot_algorithm_results(fig, axs, BFGS_state_progression, BFGS_cost_progression, BFGS_time_progression, 'BFGS')

    plt.show()
    print('pausing')


if __name__ == "__main__":
    main()







