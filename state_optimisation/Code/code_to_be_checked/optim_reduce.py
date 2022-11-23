import itertools as it
import multiprocessing as m
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize
from scipy.special import factorial

# from progressbar import ProgressBar
# pbar = ProgressBar()

ide = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
si = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)


def ham(
    alpha, n
):  # I must change this in order to allow alpha to be of different sizes
    H = (
        alpha[0] * SC_estim_ham(sx, n)
        + alpha[1] * SC_estim_ham(sy, n)
        + alpha[2] * SC_estim_ham(sz, n)
    )
    return H


def til_state(n):
    state = (
        qt.tensor([d1[0]] * n)
        + qt.tensor([d1[1]] * n)
        + qt.tensor([d2[0]] * n)
        + qt.tensor([d2[1]] * n)
        + qt.tensor([d3[0]] * n)
        + qt.tensor([d3[1]] * n)
    )
    return state.unit()


def SC_estim_ham(pauli, N):  # sets up estimation hamiltonian for a pauli
    h = qt.tensor([qt.Qobj(np.zeros((2, 2)))] * N)
    for i in range(0, N):
        a = [si] * N
        a[i] = pauli
        b = qt.tensor(a)
        h += b
    return h


def noise(n, gamma, rho):
    e0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    e1 = qt.Qobj([[0, 0], [0, np.sqrt(gamma)]])
    kraus = [x for x in it.product([e0, e1], repeat=n)]
    out = []
    for i in range(len(kraus)):
        out.append(qt.tensor([kraus[i][j] for j in range(n)]))
    state = qt.tensor([qt.Qobj(np.zeros((2, 2)))] * n)
    for i in range(len(out)):
        state += out[i] * rho * out[i]
    return state


def hamiltonian(
    alpha, rho, n
):  # I must change this in order to allow alpha to be of different sizes
    H = qt.Qobj.expm(
        -1j
        * (
            alpha[0] * SC_estim_ham(sx, n)
            + alpha[1] * SC_estim_ham(sy, n)
            + alpha[2] * SC_estim_ham(sz, n)
        )
    )
    Hdag = qt.Qobj.expm(
        1j
        * (
            alpha[0] * SC_estim_ham(sx, n)
            + alpha[1] * SC_estim_ham(sy, n)
            + alpha[2] * SC_estim_ham(sz, n)
        )
    )
    output = H * rho * Hdag
    return output


def final_state(alpha, gamma, phi, n):
    rho = phi * phi.dag()
    rho_alpha = hamiltonian(alpha, rho, n)
    rho_n = noise(n, gamma, rho_alpha)
    return rho_n


def til_povm(n):
    povms = []
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) + qt.tensor([qt.sigmax()] * n)) / 6
    )
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) - qt.tensor([qt.sigmax()] * n)) / 6
    )
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) + qt.tensor([qt.sigmay()] * n)) / 6
    )
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) - qt.tensor([qt.sigmay()] * n)) / 6
    )
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) + qt.tensor([qt.sigmaz()] * n)) / 6
    )
    povms = np.append(
        povms, (qt.tensor([qt.identity(2)] * n) - qt.tensor([qt.sigmaz()] * n)) / 6
    )
    return povms


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
        n_state = qt.Qobj(n_state, dims=[[2] * n, [1] * n])
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


def qfim(rhot, L):
    mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat[i, j] = 4 * (
                np.real(
                    qt.Qobj.tr(L[i] * L[j] * rhot)
                    - qt.Qobj.tr(L[i] * rhot) * qt.Qobj.tr(L[j] * rhot)
                )
            )
    a = np.matrix.trace(np.linalg.inv(mat))
    return a


def A(k, n):
    pauli = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    alpha = [1e-4, 2e-4, 3e-4]
    L = qt.Qobj(np.zeros((2 ** n, 2 ** n)), dims=([[2] * n, [2] * n]))
    for i in range(10):
        for j in range(10):
            L += (
                (1j ** (i + j))
                / (i + j + 1)
                * (ham(alpha, n) ** i)
                * SC_estim_ham(pauli[k], n)
                * ((-ham(alpha, n)) ** j)
                * (1 / (factorial(i) * factorial(j)))
            )
    return L


def opt_state(n, nos, itter):
    L = [0] * 3
    for i in range(3):
        L[i] = A(i, n)
    # itter            = 5
    # n                = 3
    alpha = [1e-4, 2e-4, 3e-4]
    # nos              = 20
    gamma = 0
    states = [
        qt.tensor([qt.rand_ket(2)] * n) for i in range(nos)
    ]  # [til_state(n)]*nos #[til_state(n)]*nos
    rho_t = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
    best_state = [states[0], 10000]

    for i in range(itter):
        var = [qfim(rho_t[j], L) for j in range(nos)]
        # print(var)
        for j in range(nos):
            if (
                var[j] < 0
            ):  # this is in here just to ensure that non-invertable qfim's are rejected
                states[j] = best_state[0]
                var[j] = best_state[1]
            if var[j] < best_state[1]:
                best_state[0] = states[j]
                best_state[1] = var[j]
        states = gener(states, var, 1 / 3, n)
        states[-1] = best_state[0]
        rho_t = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
        print(best_state)
    fig, ax = qt.hinton(best_state[0] * best_state[0].dag())
    plt.show()
    with open("3_para_out.txt", "w") as f:
        print(best_state, file=f)
    return best_state
