from scipy.optimize import minimize


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


def get_est(alpha, gamma, povms, phi, rho, total_measurment, n, corse):
    print(os.getpid())
    dat = data(povms, rho, total_measurment)
    estimate = estimator(alpha, gamma, povms, dat, phi, n, corse)
    return estimate


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


def qfim(phi, rhot, alpha, gamma, n):
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
