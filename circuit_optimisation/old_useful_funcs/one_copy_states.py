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
gammas        = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

gammasl  =  len(gammas)



basis2        = h.basis_gen(n)
du            = [h.hamiltonian(sx, 2), h.hamiltonian(sy, 2), h.hamiltonian(sz, 2)]
kraus         = [h.kraus_set(2,gammas[i]) for i in range(gammasl)]



options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

hcrb_res      = np.zeros(gammasl)
cfi_res       = np.zeros((gammasl))
state_cfi     = np.zeros((gammasl), dtype = object)

basisn        = h.basis_gen(2)
povms         = [qt.Qobj(qt.basis(2**(2),k)*qt.basis(2**(2),k).dag(),dims=[[2]*(2),[2]*2]) for k in range(2**(2))]

for j in range(len(gammas)):


    kwargs_cfi          = {'basisU':basis2, 'basis_rho':basisn, 'dus': du, 'povms': povms, 'E':kraus[j], 'n': 1}
    optimizer_cfi       = GlobalBestPSO(n_particles = 32, dimensions=len(basisn)+len(basis2), options=options)#, bounds=bounds)
    cost_cfi, pos_cfi   = optimizer_cfi.optimize(h.cfi_opt, iters = 500, **kwargs_cfi)


    cfi_res[j]        = cost_cfi
    state_cfi[j]      = pos_cfi


    phasest     = state_cfi[j]
    phases      = phasest[:16]
    hcrbt       = h.hcrb_single(phases, basisn, du, kraus[j])
    hcrb_res[j] = hcrbt

    np.save(os.path.join('results_one_copy', 'hcrb_res'), hcrb_res)
    np.save(os.path.join('results_one_copy', 'cfi_res'     ), cfi_res      )
    np.save(os.path.join('results_one_copy', 'cfi_states'  ), state_cfi    )


















##

