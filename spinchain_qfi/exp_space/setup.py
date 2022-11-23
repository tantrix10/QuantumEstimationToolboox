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

# Logging

# log_level = logging.INFO
# data_dir = "data"
# ****************************************************************
# Define the physics of the problem


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




def ham(static, estim, control, control2):
    return static * Hs + estim * He + control * Hc+ control2*Hc2 
