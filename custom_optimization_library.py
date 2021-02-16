import numpy as np
import sympy
import math
import matplotlib.pyplot as plt
import time


def gradient_descent(cost_function, state_sym, initial_guess, gamma=0.03, iterations=10):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get update direction, magnitude scaled by -gamma
        delta_x            = -1*gamma*eval_gradient
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history


def gradient_descent_line_search(cost_function, state_sym, initial_guess, line_search_intervals, iterations=10):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get update direction, magnitude scaled by -gamma
        cost_best = math.inf
        for ls in range(line_search_intervals.shape[0]):
            delta_x_guess      = -1*line_search_intervals[ls]*eval_gradient
            state_update_guess = state_value + delta_x_guess 
            cost_guess         = eval_expression(cost_function, state_update_guess)
            if cost_guess < cost_best:
                cost_best          = cost_guess
                state_update_best  = state_update_guess 
            else:
                break
        # Apply state update
        state_value        = state_update_best
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history


# TODO: def gradient_descent_barzilai_berwin():

def newton_type_optimization(cost_function, state_sym, initial_guess, iterations=10, enforce_PD=False):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)
    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get Hessian
        symbolic_hessian   = get_gradient(symbolic_gradient, state_sym, state_value)
        eval_hessian       = eval_expression(symbolic_hessian, state_value)
        if enforce_PD and not isPD(eval_hessian):
            eval_hessian = make_pd(eval_hessian)
        # Get update direction, magnitude scaled by -gamma
        delta_x            = -1*np.dot(np.linalg.pinv(eval_hessian),eval_gradient)
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history



def gauss_newton_optimization(cost_function, r, state_sym, initial_guess, iterations=10):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get G
        symbolic_G         = get_gradient(r, state_sym, state_value)
        eval_G             = eval_expression(symbolic_G, state_value)
        # Get update direction, magnitude scaled by -gamma
        pseudo_hessian     = np.dot(eval_G, np.transpose(eval_G))
        delta_x            = -1*np.dot( np.linalg.pinv(pseudo_hessian)   ,eval_gradient)
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history


def levenberg_marquardt_optimization(cost_function, r,  state_sym, initial_guess, lambda_value=1e-2, iterations=10):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get G
        symbolic_G         = get_gradient(r, state_sym, state_value)
        eval_G             = eval_expression(symbolic_G, state_value)
        # Get update direction, magnitude scaled by -gamma
        pseudo_hessian     = np.dot(eval_G, np.transpose(eval_G)) + lambda_value * np.eye(eval_G.shape[0])
        delta_x            = -1*np.dot( np.linalg.pinv(pseudo_hessian)   ,eval_gradient)
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history



def constant_hessian_optimzation(cost_function, hessian_val, state_sym, initial_guess, iterations=10):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # Get update direction, magnitude scaled by -gamma
        delta_x            = -1*np.dot(hessian_val, eval_gradient)
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history



def BFGS_optimization(cost_function, state_sym, initial_guess, iterations=10, enforce_PD=False):
    # Set up logging variables
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    # Set up initial settingss
    state_value = initial_guess
    initial_time = time.perf_counter()

    last_hessian = np.eye(len(state_sym))
    last_g       = np.zeros((len(state_sym)))
    last_x       = np.zeros((len(state_sym)))

    # Begin iterative process
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        # Get gradient
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        # BFGS related terms
        delta_x = np.transpose(np.array(state_value   - last_x, ndmin=2))
        delta_g = np.transpose(np.array(eval_gradient - last_g, ndmin=2))
        hessian = last_hessian +   np.dot(delta_g, np.transpose(delta_g))/(np.dot(np.transpose(delta_g), delta_x))  - np.dot(np.dot( np.dot(last_hessian, delta_x) , np.transpose(delta_x)), last_hessian)/np.dot(  np.dot(  np.transpose(delta_x) , last_hessian), delta_x)
        if enforce_PD and not isPD(hessian):
            hessian = make_pd(hessian)
        last_hessian = hessian
        last_g = eval_gradient
        last_x = state_value
        # Get update direction according to BFGS
        delta_x            = -1*np.dot(np.linalg.pinv(hessian), eval_gradient)
        # Apply state update
        state_value        = state_value + delta_x
        # Get time
        time_history[i+1]  = time.perf_counter() - initial_time

        
    # Log final cost and states
    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)
    # Return logging variables
    return state_history, cost_history, time_history













def get_gradient(function, state_sym, state_value):
    symbolic_gradient = [[0]*len(function) for _ in range(len(state_sym))]
    for x in range(len(state_sym)):
        for f in range(len(function)):
            if type(function[f]) is list:
                symbolic_gradient[x][f] = sympy.diff(function[f][0],state_sym[x])
            else:
                symbolic_gradient[x][f] = sympy.diff(function[f],state_sym[x])
    return symbolic_gradient

def eval_expression(symbolic_function, state_value):
    if type(symbolic_function[0]) is list:
        eval_function = [ [0] * len(symbolic_function[0]) for _ in range(len(symbolic_function))]
        for s1 in range(len(eval_function)):
            for s2 in range(len(eval_function[0])):
                eval_function[s1][s2] = symbolic_function[s1][s2]
                if hasattr(symbolic_function[s1][s2],'free_symbols'):
                    req_symbols = symbolic_function[s1][s2].free_symbols
                    if bool(req_symbols):
                        for s in range(len(req_symbols)):
                            set_element = req_symbols.pop()
                            symbol = str(set_element)
                            state = int(symbol[1:])-1
                            eval_function[s1][s2] = eval_function[s1][s2].subs(set_element,state_value[state])
    else:
        eval_function = [0]* len(symbolic_function)
        for s1 in range(len(eval_function)):
            eval_function[s1] = symbolic_function[s1]
            if hasattr(symbolic_function[s1],'free_symbols'):
                req_symbols = symbolic_function[s1].free_symbols
                if bool(req_symbols):
                    for s in range(len(req_symbols)):
                        set_element = req_symbols.pop()
                        symbol = str(set_element)
                        state = int(symbol[1:])-1                        
                        eval_function[s1] = eval_function[s1].subs(set_element,state_value[state])
    return np.squeeze(np.array(eval_function, dtype=np.float))




def isPD(matrix):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def make_pd(matrix):
    tmp = (matrix + matrix.T)/2
    _, s, V = np.linalg.svd(tmp)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    matrix_2 = (tmp + H) /2
    matrix_3 = (matrix_2 + matrix_2.T) /2
    if isPD(matrix_3):
        return matrix_3
    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not isPD(matrix_3):
        mineig    = np.min(np.real(np.linalg.eigvals(matrix_3)))
        matrix_3 += I * (-mineig * k**2 + spacing)
        k += 1
    return matrix_3




def make_contour_plot(cost_function, levels, delta=0.1):
    x = np.arange(-10., 10., delta)
    y = np.arange(-5., 5., delta)
    X,Y = np.meshgrid(x,y)
    cost = np.zeros((X.shape[0],X.shape[1]))

    for n1 in range(X.shape[0]):
        for n2 in range(X.shape[1]):
            cost[n1,n2] = eval_expression(cost_function, np.array([X[n1,n2], Y[n1,n2]]))

    fig, axs = plt.subplots(3,1)
    CS = axs[0].contour(X,Y,cost, levels)
    axs[0].clabel(CS, inline=1, fontsize=10)
    axs[0].set_title('Contour Plot')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')

    axs[1].set_title('Error per Iteration')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Error')

    axs[2].set_title('Error vs. Time')
    axs[2].set_xlabel('Cumulative Time')
    axs[2].set_ylabel('Error')


    return fig, axs

def plot_algorithm_results(fig, axs, state_progression, cost_progression, time_progression, identifier='default'):
    axs[0].plot(state_progression[0,:], state_progression[1,:],label=identifier)
    axs[0].legend()

    axs[1].semilogy(range(len(cost_progression)), cost_progression,label=identifier)
    axs[1].legend()

    axs[2].semilogy(time_progression, cost_progression, label=identifier)
    axs[2].legend()

    plt.draw()

    return fig, axs