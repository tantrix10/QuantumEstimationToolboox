import genetic as g
import crb
import hamiltonian as h
import qutip as qt
import numpy as np
import os
from scipy.optimize  import minimize
from genetic_hol import opt_state
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO


sx            = qt.sigmax()
sy            = qt.sigmay()
sz            = qt.sigmaz()
si            = qt.identity(2)


n             = 2
gammas        = [0.4]#[0, 0.1, 0.2, 0.3, 0.4, 0.5]

ns            = [1,2,3] #[i+1 for i in range(4)]
nsl, gammasl  = len(ns), len(gammas)

 hcrb_res      = np.zeros((nsl, gammasl))
 cfi_res       = np.zeros((nsl, gammasl))
 states_hcrb   = np.zeros((nsl, gammasl), dtype = object)
 state_cfi     = np.zeros((nsl, gammasl), dtype = object)

basis2        = h.basis_gen(n)
du            = [h.hamiltonian(sx, 2), h.hamiltonian(sy, 2), h.hamiltonian(sz, 2)]
kraus         = [h.kraus_set(2,gammas[i]) for i in range(gammasl)]



options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}



############################################################################
###### remember to increase processors in pool before running on orac ######
############################################################################


 for i in range(len(ns)):
     basisn        = h.basis_gen(2*ns[i])
     povms         = [qt.Qobj(qt.basis(2**(2*ns[i]),k)*qt.basis(2**(2*ns[i]),k).dag(),dims=[[2]*(2*ns[i]),[2]*2*ns[i]]) for k in range(2**(2*ns[i]))]
     for j in range(len(gammas)):
         phi_hcrb            = qt.Qobj(np.array(np.load(os.path.join('states_1copy_hcrb', 'gamma_'+str(gammas[j])+'.npy' ),allow_pickle=True)[0]), dims = [[2]*2,[1]*2]).unit()
         print(phi_hcrb)

         kwargs_hcrb         = {'basis':basisn, 'du': du, 'povms': povms,'phi':phi_hcrb, 'E':kraus[j], 'n': ns[i]}
         kwargs_cfi          = {'basisU':basis2, 'basis_rho':basisn, 'dus': du, 'povms': povms, 'E':kraus[j], 'n': ns[i]}

         optimizer_cfi       = GlobalBestPSO(n_particles = 32, dimensions=len(basisn)+len(basis2), options=options)#, bounds=bounds)
         cost_cfi, pos_cfi   = optimizer_cfi.optimize(h.cfi_opt, iters = 1000, **kwargs_cfi)

         optimizer_hcrb      = GlobalBestPSO(n_particles = 32, dimensions=len(basisn), options=options)#, bounds=bounds)
         cost_hcrb, pos_hcrb = optimizer_hcrb.optimize(h.hcrb_opt, iters = 1000, **kwargs_hcrb)


         hcrb_res[i,j]       = cost_hcrb*ns[i]
         cfi_res[i,j]        = cost_cfi*ns[i]
         state_cfi[i,j]      = pos_cfi
         states_hcrb[i,j]    = post_hcrb


         np.save(os.path.join('results', 'hcrb_res'    ), hcrb_res     )
         np.save(os.path.join('results', 'cfi_res'     ), cfi_res      )
         np.save(os.path.join('results', 'cfi_states'  ), state_cfi    )
         np.save(os.path.join('results', 'hcrb_states' ), states_hcrb  )

import plot




##
