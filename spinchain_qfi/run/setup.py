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
import identifiability_ctrl as idctrl

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

sx=Sx = sigmax()
sy=Sy = sigmay()
sz=Sz = sigmaz()
sigma=Sigma = [Sx, Sy, Sz]
si = Si = identity(2)

Sd = Qobj(np.array([[0, 1],
                    [0, 0]]))
Sm = Qobj(np.array([[0, 0],
                    [1, 0]]))
Sd_m = Qobj(np.array([[1, 0],
                      [0, 0]]))
Sm_d = Qobj(np.array([[0, 0],
                      [0, 1]]))
nolla = Qobj(np.array([[0, 0],
                      [0, 0]]))
u = Qobj([[1], [0]])
d = Qobj([[0], [1]])


init = tensor(u,u)
targ = (tensor( d, d)+tensor(u,u))/np.sqrt(2)




#functions for defining system Hamiltonians

def inter_ham(pauli,N):#sets up interaction hamiltonian for a  pauli
    h=tensor([Qobj(np.zeros((2,2)))]*N)
    for i in range(0,N-1):
        #print(N,i+1,i)
        a=[si]*N
        a[i]=pauli
        a[i+1]=pauli
        b=tensor(a)
        h+=b
    return h

def estim_ham(pauli,N):#sets up estimation hamiltonian for a pauli
    h=tensor([Qobj(np.zeros((2,2)))]*N)
    for i in range(0,N):
        #print(N,i)
        a=[si]*N
        a[i]=pauli
        b=tensor(a)
        h+=b
    return h

def control_ham(pauli,N):#sets up the control hamiltonians for a pauli
    h=[si]*N
    h[0]=pauli
    a=tensor(h)
    return a


# Embedding
def embedding(l):
    return tensor(sigma[l], Si)
dyn.embedding = embedding


def ham(static, estim, control, control2):
    return static * Hs + estim * He + control * Hc+ control2*Hc2 




def bloch(delt, amp, ind):
    """
    Computes the final bloch vector components depending on a parameter and
    control pulse
    """
    state = dyn.initial
    for l in range(n_ts):
        H1 = ham(1, delt, amp[l])
        evo_1 = (-(dyn.evo_time / n_ts) * 1j * H1).expm()
        state = evo_1 * state
    dmatr = state.ptrace(0)
    bz = np.real((dmatr * sigma[ind]).tr())
    return bz


def prob_z(delt, amp, ind):
    """
    Computes the final bloch vector components depending on a parameter and
    control pulse
    """
    state = dyn.initial
    res1 = np.empty(n_ts+1)
    res2 = np.empty(n_ts+1)
    res1[0] = (1 + np.real((state.ptrace(0) * sigma[ind]).tr())) / 2
    res2[0] = (1 + np.real((state.ptrace(N-1) * sigma[ind]).tr())) / 2
    for l in range(n_ts):
        H1 = ham(1, delt, amp[l])
        evo_1 = (-(dyn.evo_time / n_ts) * 1j * H1).expm()
        state = evo_1 * state
        res1[l+1] = (1 + np.real((state.ptrace(0) * sigma[ind]).tr())) / 2
        res2[l+1] = (1 + np.real((state.ptrace(N-1) * sigma[ind]).tr())) / 2
    return res1, res2




def run(n, chain_type, tot_time, time_steps):
    """
    n         : number of qubits
    chain_type: heisenberg or XY couples spin chian
    tot_time  : total evolution time
    time_steps: number of linear time steps 


    """
    global Hs, He, Hc, Hc2


    #set up system Hamiltonians 

    if chain_type == 'heis':
        Hs=0.5*(inter_ham(sx,n)+inter_ham(sy,n)+inter_ham(sz,n))
    elif chain_type == 'XY':
        Hs=0.5*( inter_ham(sx,n) + inter_ham(sy,n))
    else:
        print('pick heis or XY')

    He  = estim_ham(sz,n)
    Hc  = control_ham(sz,n)
    Hc2 = control_ham(sx,n)

    # parameter to be estimated around this fixed value
    delta = 1
    # drift generator at the chosen parameter value
    dyn.drift_dyn_gen = ham(1, delta, 0, 0)
    # parameter derivative (asssume linear dependence)
    dyn.drift_dir = He
    # control hamiltonian
    dyn.ctrl_dyn_gen = [Hc, Hc2]
    #dyn.ctrl_dyn_gen = [Hc2]

    #dyn.ctrl_dyn_gen=[Hc]

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
    tc.fid_err_targ = -4*(n**2)*(tot_time)
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
    pgen = pulsegen.create_pulse_gen(pulse_type='ZERO', dyn=dyn)
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
    # optim.approx_grad = True
    #        optim = optimizer.OptimizerCrabFmin(cfg, dyn)
    #        optim = optimizer.OptimizerCrab(cfg, dyn)
    #        optim = optimizer.Optimizer(cfg, dyn)
    # set up the termination conditions
    optim.termination_conditions = tc
    optim.pulse_generator = pgen

    # Generate statistics
    if True:
        # Create a stats object
        # Note that stats object is optional
        # if the Dynamics and Optimizer stats attribute is not set
        # then no stats will be collected, which could improve performance
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

    min_fid_err_idx = None
    num_init_pulses = 1
    fidelityerror = np.zeros([num_init_pulses, 1])

    p = 0

    logger.info("Setting initial pulse {}".format(p+1))
    init_amps = np.zeros([n_ts, n_ctrls])

    #print(init_amps)

    pgen.init_pulse()
    for j in range(n_ctrls):
        init_amps[:, j] = pgen.gen_pulse()
    dyn.initialize_controls(init_amps)
    dyn.compute_evolution()
    n_ts = dyn.num_tslots
    # Save initial amplitudes to a text file
    # pulsefile = "ctrl_amps_initial_" + f_ext
    # dyn.save_amps(pulsefile)
    # if (log_level <= logging.INFO):
    # print "Initial amplitudes output to file: " + pulsefile
    result = optim.run_optimization()
    fidelityerror[p, 0] = result.fid_err
    print("Terminated due to {}".format(result.termination_reason))
    return result.fid_err, np.sqrt(-result.fid_err/(4*n**2))/tot_time, result.termination_reason 
    # print("")
    # print("***********************************")
    # print("Optimising complete. Stats follow:")
    # result.stats.report()
    # print("")
    # print("Final evolution")
    # print(result.evo_full_final)

    # print("")
    # print("********* Summary *****************")
    # print("Final fidelity error {}".format(result.fid_err))
    # print("Terminated due to {}".format(result.termination_reason))
    # print("Number of iterations {}".format(result.num_iter))

    # print("Completed in {} HH:MM:SS.US".format(
    # datetime.timedelta(seconds=result.wall_time)))
    # print("Final gradient normal {}".format(result.grad_norm_final))
    # print("***********************************")
    # print("Effective sensing time percentage {}".format(np.sqrt(-result.fid_err/(4*n**2))/20 )  )


if __name__ == "__main__":
    n = 2
    chain_type = 'XY'
    tot_time = 2
    time_steps = 5
    run(n, chain_type, tot_time, time_steps)