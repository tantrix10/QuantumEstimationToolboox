import itertools as it

import numpy as np
import qutip as qt


def SC_estim_ham(pauli, n=None):
    """
    defines the single body hamiltonian for a magnetic fielding acting on a spin-1/2 system

    input:
        -pauli: Qobj, the pauli matrix specifying the "direction" of the Hamiltonian
        -n    : int , number of qubits of the system

        defualt:
            -n: int, number of qubits, specify if systems is not tensor of spin-1/2
    output:
        Hamiltonian: Qobj, Hamiltonian described above
    """

    if n is None:
        n = len(pauli[0].dims[0])

    si = qt.identity(2)

    # \sum_{i=0}^{n-1} si^{\otimes i} pauli si^{\otimes n-1-i}
    return sum(
        [qt.tensor([si if i != j else pauli for i in range(n)]) for j in range(n)]
    )


def noise(gamma, rho, n=None):
    """
     define a set of kraus operators for dephasing in the Pauli-Z direction
     input:
        -gamma: float, strength of dephasing gamma belongs to interval [0,1]
        -rho  : Qobj , input state before dephasing

        default:
            -n: int, number of qubits, specify if systems is not tensor of spin-1/2

    returns: 
        -list of Qobj's, a set of kraus operators for a given dephasing strength
    """

    # define the two single qubit krause operators as defined in
    # Quantum Computation and Quantum Information by Isaac Chuang and Michael Nielsen

    # number of qubits if state is tensor of spin-1/2 systems
    if n is None:
        n = len(rho.dims[0])

    e0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    e1 = qt.Qobj([[0, 0], [0, np.sqrt(gamma)]])

    # take all combinations of the above kraus operators for n-qubits
    # effectively apply the single qubit dephasing to every qubit
    kraus = [x for x in it.product([e0, e1], repeat=n)]

    # for each of the new kraus operators they must be tensor-ed together
    # this step is needed because it.product returns a list of tuples
    # qutip will only tensor a list.
    out = [qt.tensor(list(kraus_set)) for kraus_set in kraus]

    # return $$\sum_{}i=1^{2**n}E_i \rho E_i $$, as defined in Chuang and Nielsen above
    return sum(oper * rho * oper.dag() for oper in out)


def unitary(alpha, hams, rho):
    """
    Given an input density matrix, a set of coefficients and Hamiltonians, return the unitary 
    evolution corresponding to the given Hamiltonian H = \sum_{i=1}^{n= len(hams)} alpha_i * hams_i

    input:
        -alpha: list of floats, coefficients of hamiltonians
        -hams : list of Qobjs , hamiltonians for respective coeffs
        -rho  : Qobj          , input density matrix
    output:
        - Evolved state, Qobj

    TODO: Consider having a seperate method to return the unitary, then evolution, better for testing. 

    """

    U = qt.Qobj.expm(-1j * (sum(alph * ham for alph, ham in zip(alpha, hams))))
    return U * rho * U.dag()


def final_state(alpha, gamma, hams, phi):
    """
    Given an input ket state, a set of coefficients and Hamiltonians, a value of pauli-z dephasing,
    return the unitary  evolution corresponding to the given Hamiltonian 
    H = \sum_{i=1}^{n= len(hams)} alpha_i * hams_i which is then dephased by-gamma.

    input:
        -alpha: list of floats, coefficients of hamiltonians
        -gamma: float         , value of pauli-z dephasing strength
        -hams : list of Qobjs , hamiltonians for respective coeffs
        -phi  : Qobj          , input ket state
    output:
        - Evolved state, Qobj

    TODO: Generalise the noise channel to take a list of Kraus operators, in a similar way to unitary

    """
    # go from ket state to density matrix
    # TODO: add a check for ket, bra or density matrix and convert accordingly

    rho = phi * phi.dag()

    # apply unitary then noise
    # TODO: This should really just be an arbitary quantum channel with unitary and kraus wrapped up

    rho_alpha = unitary(alpha, hams, rho)
    rho_n = noise(gamma, rho_alpha)
    return rho_n


def dlambda(rhot, alpha, gamma, hams, phi, i, delta=1e-8):
    """
    Given an input ket state, a set of coefficients and Hamiltonians, a value of pauli-z dephasing,
    the unitary  evolution corresponding to the given Hamiltonian 
    H = \sum_{i=1}^{n= len(hams)} alpha_i * hams_i which is then dephased by-gamma.

    Take the derivative of the evolution with respect to the ith alpha coeff

    input:
        -alpha: list of floats, coefficients of hamiltonians
        -gamma: float         , value of pauli-z dephasing strength
        -hams : list of Qobjs , hamiltonians for respective coeffs
        -phi  : Qobj          , input ket state
    output:
        - derivative, Qobj

    """
    # five point method for numerical derivative, https://en.wikipedia.org/wiki/Numerical_differentiation#Higher-order_methods
    # TODO: allow for specification of general order of derivative when more/less accuracy is needed.
    # TODO: do some thinking/research if this can be improved

    temp_alpha1 = alpha[:]
    temp_alpha2 = alpha[:]
    temp_alpha3 = alpha[:]
    temp_alpha4 = alpha[:]

    temp_alpha1[i] += 2 * delta
    temp_alpha2[i] += delta
    temp_alpha3[i] -= delta
    temp_alpha4[i] -= 2 * delta

    t1 = final_state(temp_alpha1, gamma, hams, phi)
    t2 = final_state(temp_alpha2, gamma, hams, phi)
    t3 = final_state(temp_alpha3, gamma, hams, phi)
    t4 = final_state(temp_alpha4, gamma, hams, phi)

    out = (-t1 + (8 * t2) - (8 * t3) + t4) / (12 * delta)
    return out
