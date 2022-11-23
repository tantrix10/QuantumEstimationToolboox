import numpy as np
import qutip as qt
import itertools as it
#import multiprocessing as m
#import os
#import time
#import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
#from operator import xor

import scipy     as sci
import numpy.matlib
#import scipy.io
#import time
import crb
import genetic

ide    = qt.identity(2)
sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)

np.set_printoptions(precision=3,linewidth = np.inf)




def hamiltonian(pauli, n):
        """TODO: Docstring for hamiltonian.

        :pauli: The to local pauli matrix of choice
        :returns: the local hamiltonian for a given pauli matrix
        
        """
        return sum( [qt.tensor([si if i !=j else pauli for i in range(n) ]) for j in range(n)])


def unitary(n, alpha):
        """TODO: Docstring for unitary.

        :n: numer of qubits
        :alpha: the parameters we wish to estimate for pauli x,y,z collective operators 
        :returns: The unitary for a local 3D collective spin hamiltonian

        """
        Sx, Sy, Sz = [hamiltonian(sx,n),hamiltonian(sy,n),hamiltonian(sz,n)]
        H          = alpha[0]*Sx + alpha[1]*Sy + alpha[2]*Sz
        U          = qt.Qobj.expm(-1j*H)
        return U
    

#makes life easier to define the paulis and identity here
ide = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

#this is just a set of useful functions that are used through the calculation

def kron(lis,a): # returns the kronencher product of a list of matricies. a = len(list)
    if a == 1:
        out = np.kron(lis[0],lis[1])
    else:
        out = np.kron(kron(lis, a-1),lis[a])
    return out

def pauli_i(phase): # returns a single qubit rotation from a list of three phases
	lis = qt.Qobj.expm(-1j*(phase[0]*sx + phase[1]*sz + phase[2]*sy))
	return lis

def cnot_big(n, indexes): #takes the generated indexes[a, b,c] where c referes to cnot being on or off, a is the control and b is the target. n is the numer of qubits of the system
	if indexes[2] == 0:
		return qt.tensor([ide for i in range(n)])
	else:
		return qt.cnot(n, indexes[0]-1, indexes[1]-1)

def matrix_mult(lis): #takes a list a matricies and multiplies them all together
	a = lis[0]
	if len(lis) > 1:
		for i in range(len(lis)-1):
			a = lis[i+1]*a
	return a



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



#here we start to deal with the circuit itself

class circuit: #generates circuit cells from http://cds.cern.ch/record/931003/files/0602174.pdf . n is the number of qubits
    def __init__(self, n, g, du, gamma, kraus, init = False):
        self.g = g
        self.n = n
        self.gamma  = gamma
        self.kraus = kraus
        if init == False:
            self.sq  =  [np.random.choice(2) for i in range(g*n*(n+1))]
            self.tq  =  [np.random.choice(2) for i in range(g*n*(n-1))]
            self.sqp =  [2*np.pi*np.random.random() for i in range(3*sum(self.sq))]
        else:
            self.sq, self.tq = init[0], init[1], 
            self.sqp =  [2*np.pi*np.random.random() for i in range(3*sum(self.sq))]

        #size of sq and sqp is 2*g*n*(n+1)
        #size of tq is g*n*(n-1)
        
        self.cnots = []
        
        count = 0 
        for i in range(g):
            for j in range(n):
                temp_c = qt.tensor([si]*n)
                for k in range(n):
                    if k != j:
                        #print(self.tq[count])
                        if self.tq[count] == 1:
                            #print('cnot',qt.cnot(self.n, j, k))
                            temp_c = qt.cnot(self.n, j, k)*temp_c
                            count += 1
                            self.cnots.append(temp_c)
                        else:
                            self.cnots.append(temp_c)
                            count += 1
        x0 = [np.random.random() for i in range(len(self.sqp))]
        if self.gamma == 0:
            res = minimize(circuit.func, x0, args=(self,du), method = "Nelder-Mead")
        else:
            res = minimize(circuit.func_noise, x0, args=(self,du))#, method = "Nelder-Mead")
        self.qfi = res.fun
        self.sqp = res.x
        print(res.message, res.status)
        #print(self.cnots)
    
    def unitary(self):
        count = 0
        c_count = 0
        p_c = 0
        phases = np.array_split(self.sqp, sum(self.sq)) #fix this if sum(self.sq) = 0
        #print(len(self.sq),sum(self.sq), self.sq)
        #print(phases)
        u = qt.tensor([si]*self.n)
        for i in range(self.g):
            for j in range(self.n+1):
                squt = []
                for k in range(self.n):
                    if self.sq[count] == 1:
                        #print(count)
                        h = phases[p_c][0]*sx + phases[p_c][1]*sy +phases[p_c][2]*sz
                        squt.append(qt.Qobj.expm(-1j*h))
                        p_c += 1
                    else:
                        squt.append(si)
                        #count += 1
                    count += 1
                    
                if j == self.n:
                    #print('final')
                    u = qt.tensor(squt)*u
                else:
                    #print(c_count)
                    #print(self.cnots[c_count],qt.tensor(squt))
                    u = self.cnots[c_count]*qt.tensor(squt)*u
                    c_count += 1
        return u

    


def pauli_i(phase, theta= 'pi'):
    if theta == 'pi':
        theta = np.pi
    lis = qt.Qobj.expm(-1j*(theta/2)*(phase[0]*sx + phase[1]*sz + phase[2]*sy))
    return lis

"""
n  = 2
gamma = 0.5
kraus = kraus_set(n, gamma)
du = [hamiltonian(sx, n),hamiltonian(sy, n),hamiltonian(sz, n)]
sq  =  [1 for i in range(n*(n+1))]
tq  =  [1 for i in range(n*(n-1))]
a  = circuit(n,1,du, gamma, kraus, init = [sq,tq]) 
u  = a.unitary()

a.qfi
"""


