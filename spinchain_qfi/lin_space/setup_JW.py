


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

# -*- coding: utf-8 -*-
"""
@author: Jukka Kiukas
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

Optimiser for the identifiability of an unknown parameter in the dynamics
generator.
"""
import sys
# import random
# import os
import numpy as np
# import numpy.matlib as mat
import matplotlib.pyplot as plt
import datetime
# import scipy.linalg as la
# QuTiP control modules

import qutip.control.optimconfig as optimconfig
# import qutip.control.dynamics as dynamics

import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
# import qutip.control.errors as errors
# import qutip.control.fidcomp as fidcomp
# import qutip.control.propcomp as propcomp
import qutip.control.pulsegen as pulsegen

# QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from scipy.misc import derivative
import qfi_JW as idctrl

import qutip.logging_utils as logging
logger = logging.get_logger()

name = 'QFI_qubit_ctrl'

# Logging
log_level = logging.INFO  # logging.INFO
logger.setLevel(log_level)

# log_level = logging.INFO
# data_dir = "data"
# ****************************************************************
# Define the physics of the problem

cfg = optimconfig.OptimConfig()
cfg.log_level = log_level
dyn = idctrl.DynamicsSysIdUnitary(cfg)  # the dynamics object


np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180



def ham(static, estim, control, control2, Hs, He, Hc, Hc2):
    return static * Hs + estim * He + control * Hc+ control2*Hc2 



def inter_ham(n):
    M = np.zeros((2*(n+1),2*(n+1)))
    for i in range(2,n+1):
        M[i, i-1] = M[i-1, i ] = 1
    for i in range(n+3, 2*n+2):
        M[i - 1, i] = M[i, i - 1] = -1
    
    return Qobj(M)

def estim_ham(n):
    M = np.zeros((2*(n+1),2*(n+1)))

    M[1,1] = -1
    M[n + 2, n + 2] =  1

    for i in range(2,n+1):
        M[i,i] += -1

    for i in range(n+3, 2*n+2):
        M[i, i] += 1

    return Qobj(M)

def control_hamx(n):
    M = np.zeros((2*(n+1),2*(n+1)))
    M[0, 1] = M[1, 0] = M[n + 2, 0] = M[0, n + 2] =  (1/2)
    M[n + 1, 1] = M[1, n + 1] = M[n + 1, n + 2] = M[n + 2, n + 1] = -(1/2)
    return Qobj(M)

def control_hamz(n):
    M = np.zeros((2*(n+1),2*(n+1)))

    M[1,1] = -1
    M[n + 2, n + 2] = 1
    return Qobj(M)

def run(n, tot_time, time_steps):
    """
    n         : number of qubits
    chain_type: heisenberg or XY couples spin chian
    tot_time  : total evolution time
    time_steps: number of linear time steps 


    """
   
    He  = estim_ham(n)
    Hs  = inter_ham(n)
    Hc  = control_hamx(n)
    Hc2 = control_hamz(n)

    # parameter to be estimated around this fixed value
    lam = 1
    # drift generator at the chosen parameter value
    dyn.drift_dyn_gen = ham(1, lam, 0, 0, Hs, He, Hc, Hc2)
    # parameter derivative (asssume linear dependence)
    dyn.drift_dir = He
    # control hamiltonian
    dyn.ctrl_dyn_gen = [Hc, Hc2]

    init =  Qobj(np.ones((2*(n+1))))
    targ =  Qobj(np.ones((2*(n+1))))

    # Initial state
    dyn.initial = init
    dyn.fid_scale = 1.0

    dyn.target = targ  # not needed
    # Number of time slots
    dyn.num_tslots = time_steps
    # Time allowed for the evolution
    dyn.evo_time = tot_time
    # this function is called, so that the num_ctrls attribute will be set
    n_ctrls = dyn.num_ctrls
    n_ts = dyn.num_tslots


    # Create the TerminationConditions instance
    tc = termcond.TerminationConditions()
    # Fidelity error target
    tc.fid_err_targ = -4*(n**2)*(tot_time**2)
    # Maximum iterations for the optisation algorithm
    tc.max_iterations = 2000
    # Maximum (elapsed) time allowed in seconds
    tc.max_wall_time = 15*60
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    tc.min_gradient_norm = 1e-8
    # Accuracy factor
    tc.accuracy_factor = 1

    # Create a pulse generator of the specified type
    # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
    pgen = pulsegen.create_pulse_gen(pulse_type='RND', dyn=dyn)

    pgen.scaling = 1.0
    pgen.offset = 0.0
    pgen.lbound = None
    pgen.ubound = None
    # If the pulse is a periodic type, then set the pulse to be one complete
    # wave
    if isinstance(pgen, pulsegen.PulseGenPeriodic):
        pgen.num_waves = 1.0

    # *****************************
    # Configure the optimiser

    #    optim = optimizer.OptimizerBFGS(cfg, dyn)
    optim = optimizer.OptimizerLBFGSB(cfg, dyn)

    optim.termination_conditions = tc
    optim.pulse_generator = pgen

    # Generate statistics
    sts = stats.Stats()
    dyn.stats = sts
    optim.stats = sts

    # Configuration summary
    if log_level <= logging.DEBUG:
        logger.debug(
            "Optimisation config summary...\n"
            "  object classes:\n"
            "    optimizer: " + optim.__class__.__name__ +
            "\n    dynamics: " + dyn.__class__.__name__ +
            "\n    tslotcomp: " + dyn.tslot_computer.__class__.__name__ +
            "\n    fidcomp: " + dyn.fid_computer.__class__.__name__ +
            "\n    propcomp: " + dyn.prop_computer.__class__.__name__ +
            "\n    pulsegen: " + pgen.__class__.__name__)
    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "Hamiltonian"
        msg += "Drift {}:\n".format(dg_name)
        msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j+1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial state:\n"
        msg += str(dyn.initial)
        logger.info(msg)

    # *****************************

    #min_fid_err_idx = None
    num_init_pulses = 1
    fidelityerror = np.zeros([num_init_pulses, 1])

    p = 0

    logger.info("Setting initial pulse {}".format(p+1))
    init_amps = np.zeros([n_ts, n_ctrls])

    pgen.init_pulse()
    for j in range(n_ctrls):
        init_amps[:, j] = pgen.gen_pulse()
    dyn.initialize_controls(init_amps)
    dyn.compute_evolution()
    n_ts = dyn.num_tslots

    result = optim.run_optimization()
    fidelityerror[p, 0] = result.fid_err
    print(result.final_amps)
    return result.fid_err
 
if __name__ == "__main__":
    n = 8
    tot_time = 100
    time_steps =17
    print(run(n, tot_time, time_steps))


steps = [1,5,10,15,20,25]
ns = [2,3,4,5,6,7,8,9,10]
ran_runs = 10

outs = []

# if __name__ == "__main__":
#     for n in ns:
#         n_temp = []
#         for step in steps:
#             temp = []
#             for i in range(ran_runs):
#                 temp.append(run(n,5,step))
#             n_temp.append(min(temp))
#         outs.append(min(n_temp))

#     print(outs)

#     np.savetxt('out.out', outs)