import numpy as np
import sympy
import matplotlib.pyplot as plt
import time


def gradient_descent(cost_function, state_sym, initial_guess, gamma=0.01, iterations=1000):

    # We require the gradient for gradient descent
    state_history = np.zeros((len(state_sym), iterations+1))
    cost_history  = np.zeros(iterations+1) 
    time_history = np.zeros(iterations+1)

    state_value = initial_guess
    initial_time = time.perf_counter()
    for i in range(iterations):
        # Record state and cost
        state_history[:,i] = state_value
        cost_history[i]    = eval_expression(cost_function, state_value)
        # Prepare state update
        symbolic_gradient  = get_gradient(cost_function, state_sym, state_value)
        eval_gradient      = eval_expression(symbolic_gradient, state_value)
        delta_x            = -1*gamma*eval_gradient
        state_value        = state_value + delta_x
        time_history[i+1]  = time.perf_counter() - initial_time

    state_history[:,i+1] = state_value
    cost_history[i+1]    = eval_expression(cost_function, state_value)

    return state_history, cost_history, time_history


def get_gradient(function, state_sym, state_value):
    symbolic_gradient = [[0]*len(function) for _ in range(len(state_sym))]
    for x in range(len(state_sym)):
        for f in range(len(function)):
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







def make_contour_plot(cost_function, levels, delta=0.5):
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

    plt.show()

    return fig, axs