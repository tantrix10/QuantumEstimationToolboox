import qutip     as qt
import numpy     as np
import cvxpy     as cp
import itertools as it
import scipy     as sci
import numpy.matlib

ide    = qt.identity(2)
sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)
zo     = qt.Qobj([[1],[0]])
on     = qt.Qobj([[0],[1]])

#numpy.set_printoptions(threshold=np.inf)
#np.set_printoptions(precision=3,linewidth = np.inf)




def new(n,a):
    combin = [on]*n
    state = a*(qt.tensor([zo]*n))
    for i in range(n):
        temp = combin[:]
        temp[i] = zo
        temp = qt.tensor(temp)
        state += temp
    return state.unit()


def GHZ_3D(n): #returns the superposition of the tensor product of eigenvectors of all pauli matricies
    state = qt.tensor([d1[0]]*n) + qt.tensor([d1[1]]*n) +qt.tensor([d2[0]]*n) + qt.tensor([d2[1]]*n) +qt.tensor([d3[0]]*n) + qt.tensor([d3[1]]*n)
    return state.unit()


def ham(alpha, n): # Returns the hamiltonian generator of the external single body magnetic field
    H  = (alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n))
    return H


def SC_estim_ham(pauli, N):#sets up estimation hamiltonian for a given pauli (or particular mag field direction)
    h = qt.tensor([qt.Qobj(np.zeros((2,2)))]*N)
    for i in range(0,N):
        a    = [si]*N
        a[i] = pauli
        b    = qt.tensor(a)
        h    += b
    return h

def noise(n, gamma, rho): # takes a state and returns the state dephased (pauli-Z noise) by gamma amount
    e0    = qt.Qobj([[1, 0], [0, np.sqrt(1-gamma)]])
    e1    = qt.Qobj([[0,0],[0, np.sqrt(gamma)]])
    kraus =[x for x in it.product([e0,e1], repeat = n)]
    out   =[]
    for i in range(len(kraus)):
        out.append(qt.tensor([kraus[i][j] for j in range(n) ]))
    state = qt.tensor([qt.Qobj(np.zeros((2,2)))]*n)
    for i in range(len(out)):
        state += out[i]*rho*out[i]
    return state

def U_mag(alpha, rho, n): # this states a state and returns that state after experiancing an external magnetic field
    U      = qt.Qobj.expm(-1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    #Udag   = qt.Qobj.expm(1j*(alpha[0]*SC_estim_ham(sx,n)+alpha[1]*SC_estim_ham(sy,n)+alpha[2]*SC_estim_ham(sz,n)))
    output = U*rho*U.dag()
    return output

def final_state(alpha, gamma, phi, n): #returns a given state which has undergone a magnetic field and then pauli-z dephasing
    
    rho       = phi*phi.dag()
    rho = (rho.dag()+rho)/2
    rho = rho/qt.Qobj.tr(rho)
    rho_alpha = U_mag(alpha, rho, n)
    rho_n     = noise(n, gamma, rho_alpha)
    return rho_n


def dlambda(rhot,alpha, gamma, phi, n, i):
    delta = 1e-8
    temp_alpha1    = alpha[:]
    temp_alpha2    = alpha[:]
    temp_alpha3    = alpha[:]
    temp_alpha4    = alpha[:]
    temp_alpha1[i] += 2*delta
    temp_alpha2[i] += delta
    temp_alpha3[i] -= delta
    temp_alpha4[i] -= 2*delta
    t1             = final_state(temp_alpha1, gamma, phi, n)
    t2             = final_state(temp_alpha2, gamma, phi, n)
    t3             = final_state(temp_alpha3, gamma, phi, n)
    t4             = final_state(temp_alpha4, gamma, phi, n)
    out              = (-t1 + (8*t2) - (8*t3) + t4)/(12*delta)

    return out


def rank(phi, D,gamma,n):
    tol = 1e-9
    
    rank = 0
    
    if gamma == 0:
        return [np.array([1]),1]
    else:
        for i in range(2**n):
            if abs(phi[i][0][0]) != 0:
                rank +=1
    
        Dnonzero = np.delete(D,range(rank,2**n))

        return Dnonzero, rank



def SmatRank(snonzero,d, rnk, dim):

    mask    = np.triu(np.ones([rnk], dtype= bool),1)
    scols = np.zeros((rnk,rnk))
    for i in range(rnk):
        for j in range(rnk):
            scols[i,j] = np.real(snonzero[j])

    srows   = scols.transpose()
    siminsj = -srows + scols
    siplsj  = scols + srows

    diagS   = np.concatenate((snonzero.transpose(),siplsj[mask].transpose(),siplsj[mask].transpose(),np.matlib.repmat(snonzero.transpose(),2*(d-rnk),1).flatten() ))

    Smat = sci.sparse.spdiags(diagS,0,dim,dim)
    Smat = sci.sparse.csr_matrix(Smat)
    Smat = Smat.todense()
    Smat = np.matrix(Smat, dtype = complex)

    if rnk != 1:
        offdRank = 1j* sci.sparse.spdiags(siminsj[mask],0,int((rnk**2-rnk)/2),int((rnk**2-rnk)/2)).todense()
    else:
        offdRank = 0

    offdKer = -1j*sci.sparse.spdiags(np.matlib.repmat(snonzero,d-rnk,1).flatten(),0,rnk*(d-rnk),rnk*(d-rnk)).todense()   



    Smat[int(rnk+(rnk**2-rnk)/2):int(rnk+(rnk**2-rnk))                        ,int(rnk):int(rnk+(rnk**2-rnk)/2)]                                      =-offdRank;
    Smat[int(rnk):int(rnk+(rnk**2-rnk)/2)                                     ,int(rnk+(rnk**2-rnk)/2):int(rnk+(rnk**2-rnk))]                         = offdRank;
    Smat[int(rnk+(rnk**2-rnk)+rnk*(d-rnk)):int(rnk+(rnk**2-rnk)+2*rnk*(d-rnk)),int(rnk+(rnk**2-rnk)):int(rnk+(rnk**2-rnk)+rnk*(d-rnk))]               =-offdKer;
    Smat[int(rnk+(rnk**2-rnk)):int(rnk+(rnk**2-rnk)+rnk*(d-rnk))              ,int(rnk+(rnk**2-rnk)+rnk*(d-rnk)):int(rnk+(rnk**2-rnk)+2*rnk*(d-rnk))] = offdKer;

    return Smat

def tidyV(V,n):
    out = [0]*2**n
    for i in range(2**n):

        out[i] = [V[i][j][0][0] for j in range(2**n)]

    return qt.Qobj(np.concatenate([out],axis=0), dims= [[2]*n,[2]*n])



def Rmat(S):
    tol    = 1e-8
    d,v    = np.linalg.eig(S)
    ind    = d.argsort()[::-1]
    v      = v[:,ind]
    d      = d[ind]
    rank   = sum([d[i]>tol for i in range(len(d))])
    dclean = d[:rank]
    vclean = v[:,:rank]
    dout   = np.diag(dclean)

    return np.sqrt(dout)@ vclean.transpose().conjugate()




def naghol_spd(phi, alpha, gamma, n):
    #rho = (rho.dag()+rho)/2
    #rho = rho/qt.Qobj.tr(rho)
    rho   = final_state(alpha, gamma, phi, n)
    drho  = [dlambda(rho,alpha, gamma, phi, n, i) for i in range(3)]
    d    = 2**n
    npar = 3

    D, Vi = np.linalg.eigh(rho.full())

    D  = np.real(D)
    Vi = Vi[:,::-1]
    D  = D[::-1]
    Vi = qt.Qobj(Vi,dims= [[2]*n,[2]*n])

    snonzero, rnk = rank(phi,D,gamma,n)

    maskDiag = np.diag(np.ndarray.flatten(np.concatenate((np.ones([rnk,1],dtype = bool),np.zeros([d-rnk,1],dtype = bool)))))
    maskRank = np.concatenate((np.concatenate((np.triu(np.ones(rnk,dtype = bool),1), np.zeros([rnk,d-rnk],dtype = bool)),axis = 1),np.zeros([d-rnk,d],dtype = bool)))
    maskKern = np.concatenate((np.concatenate((np.zeros([rnk,rnk],dtype = bool),np.ones([rnk,d-rnk],dtype = bool)),axis = 1),np.zeros([d-rnk,d],dtype = bool)))

    fulldim = 2*rnk*d-rnk**2

    drhomat = np.zeros((fulldim,npar),dtype=np.complex_)

    for i in range(npar):
        drho[i] = (drho[i].dag()+drho[i])/2
        eigdrho = (Vi.dag())*drho[i]*Vi
        eigdrho = eigdrho.full()
        ak = eigdrho[maskKern]
        ak = ak.reshape((rnk,d-rnk)).transpose()
        ak = ak.reshape((rnk*(d-rnk)))

 
        row = np.concatenate((eigdrho[maskDiag],np.real(eigdrho[maskRank]),np.imag(eigdrho[maskRank]),np.real(ak),np.imag(ak)))
        drhomat[:,i] =  row

    S   = SmatRank(snonzero,d, rnk, fulldim)
    S   = (S.transpose().conjugate()+S)/2
    R   = Rmat(S)
    
    effdim = R.shape[0]
    idd = np.diag(np.ndarray.flatten(np.concatenate((np.ones((rnk)),2*np.ones((fulldim-rnk))))))

    V = cp.Variable((npar,npar),PSD =True)
    X = cp.Variable((fulldim,npar))

    A = cp.vstack([ cp.hstack([V , X.T @ R.conjugate().transpose()]) , cp.hstack([R @ X , np.identity(effdim) ])])

    constraints = [ cp.vstack([   cp.hstack([cp.real(A),-cp.imag(A)]), cp.hstack([cp.imag(A),cp.real(A)])   ]) >> 0,  
                                 X.T @ idd @ drhomat == np.identity(3)]

    obj = cp.Minimize(cp.trace(V))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.SCS)#, verbose = True)
    out = prob.value
    return out






#
