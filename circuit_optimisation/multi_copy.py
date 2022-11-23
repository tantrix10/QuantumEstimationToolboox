import os
import crb
#import genetic     as g
import hamiltonian as h
import pyswarms    as ps
import qutip       as qt
import numpy       as np
from   scipy.optimize              import minimize
#from   genetic_hol                 import opt_state
from   pyswarms.single.global_best import GlobalBestPSO


#######################################################################################
"""
this function states a set of states over noise and calculates the optimal cfi for
those states over multiple copies.

"""
#######################################################################################


sx            = qt.sigmax()
sy            = qt.sigmay()
sz            = qt.sigmaz()
si            = qt.identity(2)


ns            = [3]
gammas        = [0.1]

gammasl       =  len(gammas)
nsl           =  len(ns)



du            = [h.hamiltonian(sx, 2), h.hamiltonian(sy, 2), h.hamiltonian(sz, 2)]
kraus         = [h.kraus_set(2,gammas[i]) for i in range(gammasl)]
phases_in     = np.load(os.path.join('results/results_one_copy', 'cfi_states.npy' ), allow_pickle = True)
basis         = h.basis_gen(2)

options       = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}


cfi_res       = np.zeros((nsl,gammasl))
state_cfi     = np.zeros((nsl,gammasl), dtype = object)

pop_size      = 10

from memory_profiler import profile
@profile
def my_func():

    for i in range(nsl):

        nsi           = ns[i]
        basisn        = h.basis_perm_invar(2, nsi)
        povms         = [qt.Qobj(qt.basis(2**(2*nsi),k)*qt.basis(2**(2*nsi),k).dag(),dims=[[2]*(2*nsi),[2]*(2*nsi)]) for k in range(2**(2*nsi))]

        for j in range(gammasl):


            phases    = phases_in[j][:16]
            H         = sum([phases[i]*basis[i] for i in range(len(basis)) ])
            U         = qt.Qobj.expm(-1j*H)

            phi       = qt.Qobj(qt.basis(4,0), dims = [[2,2],[1,1]])
            phi_0     = U*phi
            rhotemp   = phi_0*phi_0.dag()
            rho       = sum([kraus[j][k]*rhotemp*kraus[j][k].dag()  for k in range(4) ])
            rhon      = qt.tensor([rho]*nsi)
            dphi      = [-1j*sum([ kraus[j][k]*(du[q]*rhotemp-rhotemp*du[q].dag())*kraus[j][k].dag() for k in range(4) ]) for q in range(3)]
            dphi_big  = [sum( [qt.tensor([rho if i !=j else dphi[k] for i in range(nsi) ]) for j in range(nsi)]) for k in range(3)]


            kwargs_cfi          = {'rho_0':rhon,'basisn':basisn, 'dphi':dphi_big, 'povms': povms,  'n': nsi}
            optimizer_cfi       = GlobalBestPSO(n_particles = pop_size, dimensions=len(basisn), options=options)#, bounds=bounds)
            cost_cfi, pos_cfi   = optimizer_cfi.optimize(h.cfi_multi, iters = 100, **kwargs_cfi)


            cfi_res[i,j]        = nsi*cost_cfi
            state_cfi[i,j]      = pos_cfi


            #np.save(os.path.join('results/results_multi_copy', 'cfi_res_1_copy_cfi_opt_c_copy_3copy_only'     ), cfi_res      )
            #np.save(os.path.join('results/results_multi_copy', 'cfi_povms_1_copy_cfi_opt_c_copy_3copy_only'   ), state_cfi    )
    return cfi_res, state_cfi

if __name__=='__main__':
    my_func()

















##
