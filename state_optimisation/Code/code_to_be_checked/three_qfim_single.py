import itertools as it
import multiprocessing as m
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize

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


def data(povms, rho, measurements):
    # returns probalities for each povm, see born rule
    num = len(povms)
    prob_f = np.zeros(num)
    for i in range(num):
        prob = np.real(qt.Qobj.tr(povms[i] * rho))
        b = np.random.binomial(
            1, prob, measurements
        ).tolist()  # I should remember to stick this in a loop, check if it's one and just add to the count, saves making HUGE arrays
        prob_f[i] = b.count(1) / measurements
    return prob_f


def likelyhood_func(alpha, gamma, povm, data, phi, n, corse):
    """
    other[0] = gamma
    other[1] = povm
    other[2] = data
    other[3] = phi
    other[4] = n
    other[5] = corse
    """
    like = 0
    for i in range(len(data)):
        b = np.real(qt.Qobj.tr(povm[i] * final_state(alpha, gamma, phi, n)))
        if b != 0:
            like += data[i] * np.log(b)
    return -like


def estimator(alpha, gamma, povm, data, phi, n, corse):
    x0 = [0.9, 1.9, 2.9]
    gap = 0.2
    the_bounds = [(1 - gap, 1 + gap), (2 - gap, 2 + gap), (3 - gap, 3 + gap)]
    solution = minimize(
        likelyhood_func,
        x0,
        args=(gamma, povm, data, phi, n, corse),
        method="L-BFGS-B",
        bounds=the_bounds,
    )
    # print(solution.x)
    return solution.x


def fisher_info(rhot, phi, povms, measurements, alpha, gamma, n):
    # returns classical fisher info matrix
    # expression used  https://arxiv.org/pdf/0812.4635.pdf
    delta = 1e-7
    pi_s = data(povms, rhot, measurements)
    # print(pi_s)
    pi_d = np.zeros((len(alpha), len(povms)))
    for i in range(len(alpha)):
        temp_alpha1 = alpha[:]
        temp_alpha2 = alpha[:]
        temp_alpha3 = alpha[:]
        temp_alpha4 = alpha[:]
        temp_alpha1[i] += 2 * delta
        temp_alpha2[i] += delta
        temp_alpha3[i] -= delta
        temp_alpha4[i] -= 2 * delta
        t1 = data(povms, final_state(temp_alpha1, gamma, phi, n), measurements)
        t2 = data(povms, final_state(temp_alpha2, gamma, phi, n), measurements)
        t3 = data(povms, final_state(temp_alpha3, gamma, phi, n), measurements)
        t4 = data(povms, final_state(temp_alpha4, gamma, phi, n), measurements)
        a = (-t1 + (8 * t2) - (8 * t3) + t4) / (12 * delta)
        for j in range(len(a)):
            pi_d[i, j] = a[j]

    fisher_mat = np.zeros((len(alpha), len(alpha)))

    for i in range(len(povms)):
        if pi_s[i] > 1e-15:
            fisher_mat += (np.outer(pi_d[:, i], pi_d[:, i])) / pi_s[i]
        else:
            continue

    covar = np.linalg.inv(fisher_mat)
    covar_trace = np.trace(covar)

    return covar_trace


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


def get_est(alpha, gamma, povms, phi, rho, total_measurment, n, corse):
    print(os.getpid())
    dat = data(povms, rho, total_measurment)
    estimate = estimator(alpha, gamma, povms, dat, phi, n, corse)
    return estimate


def dlambda(rhot, alpha, gamma, phi, n, i):
    delta = 1e-8
    """
    alpha_t = alpha[:]
    alpha_t[i] += delta
    rho_del = final_state(alpha_t, gamma, phi, n)
    out = (rhot-rho_del)/(delta)
    """

    temp_alpha1 = alpha[:]
    temp_alpha2 = alpha[:]
    temp_alpha3 = alpha[:]
    temp_alpha4 = alpha[:]
    temp_alpha1[i] += 2 * delta
    temp_alpha2[i] += delta
    temp_alpha3[i] -= delta
    temp_alpha4[i] -= 2 * delta
    t1 = final_state(temp_alpha1, gamma, phi, n)
    t2 = final_state(temp_alpha2, gamma, phi, n)
    t3 = final_state(temp_alpha3, gamma, phi, n)
    t4 = final_state(temp_alpha4, gamma, phi, n)
    out = (-t1 + (8 * t2) - (8 * t3) + t4) / (12 * delta)

    return out


def qfi(rhot, alpha, gamma, phi, n, i):
    D, V = qt.Qobj.eigenstates(rhot)
    L = qt.tensor([qt.Qobj(np.zeros((2, 2)))] * n)
    a = dlambda(rhot, alpha, gamma, phi, n, i)
    rank = 0
    for i in range(2 ** n):
        if abs(phi[i][0][0]) != 0:
            rank += 1
    D[:rank] = 0
    D = qt.Qobj(D)
    # D = D.tidyup(atol=1e-7)
    for i in range(2 ** n):
        for j in range(2 ** n):
            if D[i][0][0] + D[j][0][0] == 0:
                continue
            rd = (
                (V[i].dag()) * a * V[j]
            )  # speed up by putting this outside the loop, but do this when it's working
            L += ((rd / (D[i][0][0] + D[j][0][0]))) * (V[i] * V[j].dag())
    L = 2 * L
    # d = qt.Qobj.tr(a * L)
    # d = d.real
    return L


def qfim(phi, rhot, alpha, gamma, rho, n):
    L = [0] * 3
    mat = np.zeros((3, 3))
    for i in range(3):
        L[i] = qfi(rhot, alpha, gamma, phi, n, i)
    for i in range(3):
        for j in range(3):
            # print(np.real(qt.Qobj.tr(L[i]*L[j]*rhot)))
            mat[i, j] = np.real(qt.Qobj.tr(L[i] * L[j] * rhot))
    a = np.matrix.trace(np.linalg.inv(mat))
    return a


def opt_state(n, itter):
    # itter            = 5
    # n                = 3
    alpha = [1e-4, 2e-4, 3e-4]
    nos = 10
    gamma = 0
    states = [qt.tensor([qt.rand_ket(2)] * n) for i in range(nos)]  # [til_state(n)]*nos
    rho_t = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
    povms = til_povm(n)
    best_var = [0] * itter
    best_states = [0] * itter
    best_state = [states[0], 10000]
    pool = m.Pool(processes=28)

    for i in range(itter):
        var = [0] * nos
        for j in range(nos):
            var[j] = qfim(
                states[j], rho_t[j], alpha, gamma, states[j] * states[j].dag(), n
            )
        # print(var)
        for j in range(nos):
            if var[j] < 0:
                states[j] = best_state[0]
                var[j] = best_state[1]
            if var[j] < best_state[1]:
                best_state[0] = states[j]
                best_state[1] = var[j]
        best_var[i] = np.min(var)
        best_states = np.append(best_states, states[np.argmin(var)])
        states = gener(states, var, 1 / 2 ** n, n)
        states[-1] = best_state[0]
        rho_t = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
        print(best_state)
    fig, ax = qt.hinton(best_state[0] * best_state[0].dag())
    plt.show()
    with open("3_para_out.txt", "w") as f:
        print(best_state, file=f)
    return best_state


# plt.plot(all_var)
# plt.show()
