import qutip as qt

ide = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
si = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)


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


def til_state(n):
    """
    generate the 3D-GHZ state as described https://arxiv.org/abs/1507.02956 (equation 14 where \delta_i = 0 \forall i \in {1,2,3})
    input:
        -n: int, number of qubits of systems
    returns:
        -3D-GHZ: Qobj, of the corresponding ket state for n-qubits (normalised).
    """
    state = (
        qt.tensor([d1[0]] * n)
        + qt.tensor([d1[1]] * n)
        + qt.tensor([d2[0]] * n)
        + qt.tensor([d2[1]] * n)
        + qt.tensor([d3[0]] * n)
        + qt.tensor([d3[1]] * n)
    )
    return state.unit()
