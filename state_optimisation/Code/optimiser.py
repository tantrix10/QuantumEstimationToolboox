import itertools as it
import multiprocessing as m
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize


def gener(states, variences, mutatation_rate, n):
    cost = [1 / variences[i] for i in range(len(variences))]  # need to check for zeros
    cost_sum = sum(cost)
    mating_pool_relative = [
        round(((cost[i]) / cost_sum) * 100) for i in range(len(variences))
    ]
    mating_pool = []
    new_states = []
    for i in range(len(mating_pool_relative)):
        mating_pool = np.append(mating_pool, [i] * int(mating_pool_relative[i]))
    for i in range(len(states)):
        state1 = states[int(np.random.choice(mating_pool))]
        state2 = states[int(np.random.choice(mating_pool))]
        splice_point = np.random.randint(2 ** n)
        n_state = np.append(state1[:splice_point], state2[splice_point:])
        n_state = mutate(n_state, mutatation_rate, n)
        n_state = qt.Qobj(n_state).unit()
        n_state = qt.Qobj(n_state, dims=[[2 for i in range(n)], [1 for j in range(n)]])
        new_states = np.append(new_states, n_state)
    return new_states


def mutate(state, mutatation_rate, n):
    possibilities = [1, 2, 3]  # substitution, insertion, deletion
    for i in range(
        len(state)
    ):  # for each element in the state there is the same chance of a mutation occuring
        chance = np.random.random()
        if chance < mutatation_rate:
            mut = np.random.choice(
                possibilities
            )  # if a mutation does take place, then pick randomly one of three or four options
            if mut == 1:
                state[i] += (
                    2 * np.random.random() - 1 + (2 * np.random.random() - 1) * 1j
                )
            elif mut == 2:
                ele = np.random.choice(2 ** n)
                state[ele] = state[i]
            elif mut == 3:
                state[i] = 1e-8
    return state


def opt_state(
    n,
    itter,
    cost_func,
    final_state,
    init_state,
    alpha,
    hams,
    nos,
    gamma,
    mutatation_rate=None,
    verbose=True,
    save=True,
):
    """ 
    Optimises input ket for 3D-magnetometry sensing as described in https://arxiv.org/abs/1507.02956

    Algorithm described in my thesis + readme! Thesis published ASAP, readme also written ASAP. 

    input:
        -n          : int               , number of qubits
        -itter      : int               , max number of itterations
        -cost_func  : function reference, cost function to be evaluated (for now qfim in cost_func.py)
        -final_state: function reference, evoltution function to take input ket to evolved state (for now, final_state in dynamics.py)
        -init_state : function reference, function for generating initial states (for now til_state in states_measurements.py)
        -alpha      : list of floats    , list of hamiltonian paraemters to be estimated
        -hams       : list of Qobjs     , hamiltonians for respective coeffs
        -nos        : int               , number of states for genetic mating pool
        -gamma      : float             , strength of Pauli-z

        defualt:
            -mutation_rate : float [0, 1], probability of a mutation taking place on each mating 
            -verbose       : bool        , if true print progress to terminal and generate optimal state hinton plot, if false print nothing 
            -save          : bool        , if true save optimal state, if false don't     

    output:
        -Qobj: optimal state in the given number of itterations

    """

    # define init state, their evolution and init best state list
    # n.b. the magic number init best value will be gone in OO picture
    # TODO: best_state should be number tuple, a problem for when it's OO
    # n.b. for now best_state is [best ket, best ket qfi]

    states = [init_state(n) for _ in range(nos)]
    rho_t = [final_state(alpha, gamma, hams, phi) for phi in states]
    best_state = [states[0], 10000]

    # if mutation rate not set then a 'good' rate is to have on average one per state, i.e. $$1/dim(\mathcal{H})$$

    if mutatation_rate is None:
        mutatation_rate = 1 / 2 ** n

    for _ in range(itter):
        with m.Pool(processes=m.cpu_count()) as pool:
            var = list(
                pool.starmap(
                    cost_func,
                    [
                        [state, rho, alpha, gamma, n]
                        for state, rho in zip(states, rho_t)
                    ],
                )
            )
        if verbose:
            print(var, best_state)

        # if we have found a better state, update the best state
        # TODO: optimise this, I'm sure it could be better (would come with swtiching to OO)

        if np.min(var) < best_state[1]:
            index = np.argmin(var)
            best_state = [states[index], var[index]]

        # mutate states, add current best for genetic elitism, evolve states
        # N.B. this will be handled much better when switched to proper OO

        states = gener(states, var, mutatation_rate, n)
        states[-1] = best_state[0]
        rho_t = [final_state(alpha, gamma, hams, phi) for phi in states]

    if verbose:
        fig, ax = qt.hinton(best_state[0] * best_state[0].dag())
        plt.show()

    # TODO: this save method is bad, make it smarter.

    if save:
        with open("3_para_out.txt", "w") as f:
            print(best_state, file=f)

    return best_state
