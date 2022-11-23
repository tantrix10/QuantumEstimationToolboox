
import qutip as qt 
import numpy as np
from . import setup as h

from functools import reduce
x = qt.sigmax()
y = qt.sigmay()
z = qt.sigmaz()

def U(f, g, l, n,t):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    n.b. input alpha vector for now is implicitly \alpha = (0,1, ..., 0_n+1, 0, 1, ..., 0_2n+2 )

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    #define the interaction hamiltonians
    h_interx = h.inter_ham(x, n) 
    h_intery = h.inter_ham(y, n)

    #define the control hamiltonians
    hc1 = h.control_ham(x, n)
    hc2 = h.control_ham(z, n)
    #hc3 = h.control_ham(y, n) #not used in actual control hamiltonian

    #define magnetic field hamiltonian
    h_mag = h.estim_ham(z, n)

    #define total hamiltonian and unitary
    H        = (1)*(h_interx + h_intery) + l*h_mag + f*hc1 + g*hc2
    #print('pauli hamiltonian',H)
    #print('just interaction bit',h_interx+h_intery)
    U        = qt.Qobj.expm(-1j*(1/2)*t*H)
    return U 


def D_vec(f, g, l, n):
    u = reduce(lambda x,y: x*y, [U(f[i], g[i], l, n, 1/len(f)) for i in range(len(f))])
    #define initial state
    zero     = qt.basis(2,0)
    # plus     =  (qt.basis(2,0)+qt.basis(2,1)).unit()
    # states   = [plus]
    # states.extend([zero]*(n-1))
    # in_state = qt.tensor(states)
    in_state = qt.tensor([zero]*n)
    #print(in_state)

    #time evolved state and reduced 1st state
    state_t   = u*in_state
    red_state = state_t.ptrace(0)

    #print(state_t, red_state)
    
    #define expectations 
    e_x = np.real(qt.Qobj.tr(x*red_state))
    e_y = np.real(qt.Qobj.tr(y*red_state))
    e_z = np.real(qt.Qobj.tr(z*red_state))
    #print('red state',red_state)
    #print(np.real(qt.Qobj.tr((y-1j*x)*red_state)))
    # e_x = np.real(qt.Qobj.tr(hc1*state_t*state_t.dag()))
    # e_y = np.real(qt.Qobj.tr(hc3*state_t*state_t.dag()))
    # e_z = np.real(qt.Qobj.tr(hc2*state_t*state_t.dag()))
    #print(e_x**2+e_y**2+e_z**2)
    return np.array([e_x, e_y, e_z])


def D_given_u(u,n):
    #define initial state
    zero     = qt.basis(2,0)
    # plus     =  (qt.basis(2,0)+qt.basis(2,1)).unit()
    # states   = [plus]
    # states.extend([zero]*(n-1))
    # in_state = qt.tensor(states)
    in_state = qt.tensor([zero]*n)
    #print(in_state)

    #time evolved state and reduced 1st state
    state_t   = u*in_state
    red_state = state_t.ptrace(0)

    #print(state_t, red_state)
    
    #define expectations 
    e_x = np.real(qt.Qobj.tr(x*red_state))
    e_y = np.real(qt.Qobj.tr(y*red_state))
    e_z = np.real(qt.Qobj.tr(z*red_state))
    #print('red state',red_state)
    #print(np.real(qt.Qobj.tr((y-1j*x)*red_state)))
    # e_x = np.real(qt.Qobj.tr(hc1*state_t*state_t.dag()))
    # e_y = np.real(qt.Qobj.tr(hc3*state_t*state_t.dag()))
    # e_z = np.real(qt.Qobj.tr(hc2*state_t*state_t.dag()))
    #print(e_x**2+e_y**2+e_z**2)
    return np.array([e_x, e_y, e_z])