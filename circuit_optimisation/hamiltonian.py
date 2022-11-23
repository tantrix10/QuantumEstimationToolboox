import qutip as qt
import numpy as np
import crb
import itertools as it
import multiprocessing as m



sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)

#pool = m.Pool(processes = 28)


def povms_gen(n):
    return [qt.basis(2**n, i)*qt.basis(2**n,i).dag() for i in range(2**n)]


def hamiltonian(pauli, n):
        """TODO: Docstring for hamiltonian.

        :pauli: The to local pauli matrix of choice
        :returns: the local hamiltonian for a given pauli matrix

        """
        return sum( [qt.tensor([si if i !=j else pauli for i in range(n) ]) for j in range(n)])


def basis_gen(n):
    """define a set of kraus operators for dephasing

    :n: number of qubits
    :gamma: strength of dephasing gamma belongs to interval [0,1]
    :returns: a set of kraus operators for a given dephasing strength

    """
    kraus =[x for x in it.product([si, sx, sy, sz], repeat = n)]
    out   =[]
    for i in range(len(kraus)):
            out.append(qt.tensor([kraus[i][j] for j in range(n) ]))
    return out


def basis_perm_invar(q, c):
    """
    q: number of qubits
    c: number of copies of each subsystem
    this defines a set of basis elements for q-qubits c-copies that is
    permutationally invarient on the c-copy level.
    """
    init_bas = basis_gen(q)
    combi    = [x for x in  it.combinations_with_replacement(init_bas, c)]
    perms    = [[x for x in it.permutations(combi[i], c)] for i in range(len(combi)) ]
    out = []
    for i in range(len(perms)):
        temp = perms[i]
        temp_out = []
        for j in range(len(temp)):
            temp_out.append(qt.tensor([temp[j][k] for k in range(len(temp[j])) ]))
        out.append(sum(temp_out)/len(temp_out))
    return out



def kraus_set(n,gamma):
        """define a set of kraus operators for dephasing

        :n: number of qubits
        :gamma: strength of dephasing gamma belongs to interval [0,1]
        :returns: a set of kraus operators for a given dephasing strength

        """
        e0    = qt.Qobj([[1, 0], [0, np.sqrt(1-gamma)]])
        e1    = qt.Qobj([[0,0],[0, np.sqrt(gamma)]])
        kraus =[x for x in it.product([e0,e1], repeat = n)]
        out   =[]
        for i in range(len(kraus)):
                out.append(qt.tensor([kraus[i][j] for j in range(n) ]))
        return out







def hcrb_opt(phases, basis, du, povms, phi, E, n):
    shape = np.shape(phases)[0]
    #basis, du, povms, phi, E, n = [args[i] for i in range(6)]
    #print(povms)
    rhotemp  = phi*phi.dag()
    rho      = sum([E[i]*rhotemp*E[i]  for i in range(len(E))])
    rho_0    = qt.tensor([rho]*n)
    #N-copy state that is optimal for 1-copy HCRB
    H        = [sum([phases[j][i]*basis[i] for i in range(len(basis))]) for j in range(shape)]
    U        = [qt.Qobj.expm(-1j*H[i]) for i in range(shape)]
    #print(U)
    #generates the unitary for measurement on N-copies
    rhott    = [U[i]*rho_0*U[i].dag() for i in range(shape)]
    #print(rho[1], du[1])
    dphi     = [-1j*sum([ E[j]*(du[i]*rhotemp-rhotemp*du[i].dag())*E[j].dag() for j in range(len(E)) ]) for i in range(3)]
    dphi_big = [[U[m]*sum( [qt.tensor([rho if i !=j else dphi[k] for i in range(n) ]) for j in range(n)])*U[m].dag() for k in range(3)] for m in range(shape)]
    #generates the derivatives and final state
    #print(povms[1], rhott[1])
    out =  list(pool.starmap(crb.fisher_info, [[rhott[i], dphi_big[i], povms] for i in range(shape) ] ))
    return out


def cfi_opt(phases, basisU, basis_rho, dus, povms, E, n  ):

    phases_rho, phasesU             = phases[:,:16], phases[:,16:]
    shape = np.shape(phases_rho)[0]
    #print(np.shape(phases_rho))
    #print(phases_rho[0])
    #basisU, basis_rho, du, povms, E, n = [args[i] for i in range(6)]
    #print(phases_rho, len(basisU))
    zero     = qt.Qobj(qt.basis(4,0)*qt.basis(4,0).dag(),dims = [[2,2],[2,2]]).dag()
    U_phi    = [qt.Qobj(qt.Qobj.expm(-1j*sum([phases_rho[k][i]*basisU[i] for i in range(len(basisU))])), dims = [[2,2],[2,2]]) for k in range(shape)]
    #print(U_phi, zero)
    rho      = [U_phi[i]*zero*U_phi[i].dag() for i in range(shape)]
    rho_final  =[sum([E[i]*rho[k]*E[i].dag()  for i in range(len(E))]) for k in range(shape)]


    rho_0    = [qt.tensor([rho_final[i]]*n) for i in range(shape)]
    #generates an N-copy state from a 1-copy unitary
    H        = [sum([phasesU[k][i]*basis_rho[i] for i in range(len(basis_rho))]) for k in range(shape)]
    U        = [qt.Qobj.expm(-1j*H[i]) for i in range(shape)]
    #generates the untiary for measurement
    rhot     = [U[i]*rho_0[i]*U[i].dag() for i in range(shape)]

    dphi     = [ [sum([-1j*E[j]*(dus[i]*rho[k]-rho[k]*dus[i].dag())*E[j].dag() for j in range(len(E))])
                                                                                for i in range(3)]
                                                                                    for k in range(shape)]
    dphi_big = [[U[m]*sum( [qt.tensor([rho_final[m] if i !=j else dphi[m][k] for i in range(n) ]) for j in range(n)])*U[m].dag() for k in range(3)] for m in range(shape)]
    #generates the derivatives and final state
    #print("<<<<<<<< FINISHED  >>>>>>>>>")
    out = list(pool.starmap(crb.fisher_info, [[rhot[i], dphi_big[i], povms] for i in range(shape) ] ))
    return out



def hcrb_single(phases, basis, du, E):

    H        = sum([phases[i]*basis[i] for i in range(len(basis))])
    U        = qt.Qobj.expm(-1j*H)

    phi      = qt.Qobj(qt.basis(4,0), dims = [[2,2],[1,1]])
    phi_0    = U*phi
    rhotemp  = phi_0*phi_0.dag()
    rho      = sum([E[i]*rhotemp*E[i]  for i in range(len(E))])

    dphi     = [-1j*sum([ E[j]*(du[i]*rhotemp-rhotemp*du[i].dag())*E[j].dag() for j in range(len(E)) ]) for i in range(3)]

    out =  crb.hcrb(rho, dphi, phi_0)
    return out



############################################################################################
############################################################################################

def cfi_multi(phases, rho_0,basisn, dphi,  povms, n  ):

    pool = m.Pool(m.cpu_count())
    #print(m.cpu_count())
    shape = np.shape(phases)[0]
    out = list(pool.starmap(cfi_multi_eval, [[phases[i], rho_0,basisn, dphi,  povms, n ] for i in range(shape) ] ))
    pool.close()
    return [n*out[i] for i in range(len(out))]



def cfi_multi_eval(phases, rho_0,basisn, dphi,  povms, n ):
    #shape = np.shape(phases)[0]
    #generates an N-copy state from a 1-copy unitary
    H        = sum([phases[i]*basisn[i] for i in range(len(basisn))])
    U        = qt.Qobj.expm(-1j*H)
    #generates the untiary for measurement
    rho_final     = U*rho_0*U.dag()
    dphi_big = [U*dphi[k]*U.dag() for k in range(3)]
    out = crb.fisher_info(rho_final, dphi_big, povms)
    del H, U, rho_final, dphi_big
    return out 


############################################################################################
############################################################################################


def opt_q_qubit_state(phases,basisn, kraus, dphi,  povms, n):
    pool = m.Pool(m.cpu_count())
    #print(m.cpu_count())
    shape = np.shape(phases)[0]
    out = list(pool.starmap(opt_q_eval, [[phases[i], basisn, kraus, dphi,  povms, n ] for i in range(shape) ] ))
    pool.close()

    return out


def opt_q_eval(phases, basisn, kraus, du,  povms, n):

    state, measure = phases[:2*2**n], phases[2*2**n:]
    u,v = np.split(state,2)
    phi = qt.Qobj(u+1j*v, dims = [[2]*n,[1]*n] ).unit()
    rho_0 = phi*phi.dag()

    rho       = sum([kraus[k]*rho_0*kraus[k].dag()  for k in range(len(kraus)) ])
    dphi      = [-1j*sum([ kraus[k]*(du[q]*rho_0-rho_0*du[q].dag())*kraus[k].dag() for k in range(len(kraus)) ]) for q in range(3)]



    H        = sum([measure[i]*basisn[i] for i in range(len(basisn))])
    U        = qt.Qobj.expm(-1j*H)
    #generates the untiary for measurement

    rho_final     = U*rho_0*U.dag()
    dphi_big = [U*dphi[k]*U.dag() for k in range(3)]

    return crb.fisher_info(rho_final, dphi_big, povms)



############################################################################################
############################################################################################







def opt_n_copy_state(phases, basis, basisn, kraus, du,  povms, c):
    pool = m.Pool(m.cpu_count())
    #print(m.cpu_count())
    shape = np.shape(phases)[0]
    out = list(pool.starmap(opt_n_copy_eval, [[phases[i], basis, basisn, kraus, du,  povms, c ] for i in range(shape) ] ))
    pool.close()
    return [c*out[i] for i in range(len(out))]


def opt_n_copy_eval(phases_in, basis, basisn, kraus, du,  povms, c):
    phases, measure    = phases_in[:16], phases_in[16:]
    H         = sum([phases[i]*basis[i] for i in range(len(basis)) ])
    U         = qt.Qobj.expm(-1j*H)

    phi       = qt.Qobj(qt.basis(4,0), dims = [[2,2],[1,1]])
    phi_0     = U*phi
    rhotemp   = phi_0*phi_0.dag()
    rho       = sum([kraus[k]*rhotemp*kraus[k].dag()  for k in range(4) ])
    rhon      = qt.tensor([rho]*c)
    dphi      = [-1j*sum([ kraus[k]*(du[q]*rhotemp-rhotemp*du[q].dag())*kraus[k].dag() for k in range(4) ]) for q in range(3)]
    dphi_t  = [sum( [qt.tensor([rho if i !=j else dphi[k] for i in range(c) ]) for j in range(c)]) for k in range(3)]

    #print(basisn[1])
    Hm        = sum([measure[i]*basisn[i] for i in range(len(basisn))])
    Um        = qt.Qobj.expm(-1j*Hm)
    #print(Um, dphi_t[1])
    rho_final     = Um*rhon*Um.dag()
    dphi_big      = [Um*dphi_t[k]*Um.dag() for k in range(3)]

    return crb.fisher_info(rho_final, dphi_big, povms)



############################################################################################
############################################################################################






def opt_HCRB_ncopy():
    return crb.fisher_info








############I just don;t like atom removing my space at the bottom
