#import circuit
import cvxpy     as cp
import qutip as qt
import scipy     as sci
import numpy as np


def eig(rho, phi, n):
    """eigen decompostion, assuming parameter is ~0

    :rho: evolved state
    :returns: Eigvenvectors and tidied Eigenvalues

    """
         
    D, Vi = np.linalg.eigh(rho.full())
    D     = np.real(D)
    Vi    = Vi[:,::-1]
    D     = D[::-1]
    Vi    = qt.Qobj(Vi,dims= [[2]*n,[2]*n])


    rank = sum([D[i] > 1e-7 for i in range(2**n) ]) #sum([phi[i][0][0] != 0 for i in range(2**n)  ])
    D[rank:] = 0
    Dclean = D[:rank]
    #print(rank, D, Dclean)
    return D, Vi, Dclean, rank



def func_noise(phases, *args):
    self, du = args[0], args[1]
    self.sqp = phases
    zo       = qt.basis(2)
    u        = circuit.unitary(self)
    #qf_mat   = np.zeros((3,3))
    phi      = qt.tensor([zo]*self.n)
    phi_0    = u*phi
    rho_0    = phi_0*phi_0.dag()
    rho      = sum([self.kraus[i]*rho_0*self.kraus[i] for i in range(len(self.kraus))])
    drho     = [1j*sum([self.kraus[i]*(du[j]*rho_0-rho_0*du[j])*self.kraus[i] for i in range(len(self.kraus))])  for j in range(3)]
    
    D, Vi, snonzero, rnk = circuit.eig(rho, phi_0, n)
    d     = 2**self.n
    npar  = 3
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

    S   = circuit.SmatRank(snonzero,d, rnk, fulldim)
    S   = (S.transpose().conjugate()+S)/2
    R   = sci.linalg.sqrtm(S) #Rmat(S)

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
    if type(out) == None:
        return 1e+5
    else:
        return out

def hcrb(rho, drho, phi_0):
    #self, du = args[0], args[1]
    #self.sqp = phases
    #zo       = qt.basis(2)
    #u        = circuit.unitary(self)
    #qf_mat   = np.zeros((3,3))
    #phi      = qt.tensor([zo]*self.n)
    #phi_0    = u*phi
    #rho_0    = phi_0*phi_0.dag()
    #rho      = sum([self.kraus[i]*rho_0*self.kraus[i] for i in range(len(self.kraus))])
    #drho     = [1j*sum([self.kraus[i]*(du[j]*rho_0-rho_0*du[j])*self.kraus[i] for i in range(len(self.kraus))])  for j in range(3)]
    n = 2
    D, Vi, snonzero, rnk = eig(rho, phi_0, n)
    d     = 2**n
    npar  = 3
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
    R   = sci.linalg.sqrtm(S) #Rmat(S)

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
    if type(out) == None:
        return 1e+5
    else:
        return out

def SmatRank(snonzero,d, rnk, dim):

    mask    = np.triu(np.ones([rnk], dtype= bool),1)
    scols   = np.zeros((rnk,rnk))
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



def func(phases, *args):
    self, du = args[0], args[1]
    self.sqp = phases
    zo       = qt.basis(2)
    u        = circuit.unitary(self)
    qf_mat   = np.zeros((3,3))
    phi      = qt.tensor([zo]*self.n)
    phi      = u*phi
    dphi     = [-1j*du[i]*phi for i in range(3)]
    #nag_X(phi,dphi, n)
    
    d    = 2**n
    npar = 3

    psidpsi = phi.dag()*dphi;
    pardphi = phi * psidpsi
    #print(dphi, pardphi)
    Lmat    = 2 * (dphi-pardphi)
    #print(Lmat)
    psi = np.zeros(d,dtype= complex)
    for i in range(d):
        psi[i] = phi[i][0][0]

    Lmatt = np.zeros((d,npar),dtype = complex)
    for i in range(d):
        for j in range(npar):
            Lmatt[i,j] = Lmat[j][i] 
    #print(Lmatt)
    V = cp.Variable((npar,npar),PSD =True)
    X = cp.Variable((d,npar),complex = True)

    A = cp.vstack([ cp.hstack([V , X.H ]) , cp.hstack([X , np.identity(d) ])  ])


    constraints = [ cp.vstack([   cp.hstack([cp.real(A),-cp.imag(A)]), cp.hstack([cp.imag(A),cp.real(A)])   ]) >> 0, 
                    cp.real(X.H @ Lmatt) == np.identity(3),
                     ((psi.transpose()).conjugate() @ X) == 0]

    obj = cp.Minimize(cp.trace(V))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.SCS)#, verbose = True)
    #out = prob.solution
    #print(X.value)
    #print(prob.value)
    out = prob.value
    return out


def data(rho, povms):
    #returns probalities for each povm, see born rule
    num    = len(povms)
    prob_f = np.zeros(num)
    for i in range(num):
        prob_f[i]      = np.real(qt.Qobj.tr(povms[i]*rho))
    return prob_f

def fisher_info(rhot, drho, povms):
    #returns classical fisher info matrix
    #expression used  https://arxiv.org/pdf/0812.4635.pdf
    pi_s = data(rhot, povms)
    pi_d = np.zeros((3, len(povms) ))
    for i in range(3):
        a              = data(drho[i], povms)
        for j in range(len(a)):
            pi_d[i,j] = a[j]

    fisher_mat = np.zeros((3,3))

    for i in range(len(povms)):
        if pi_s[i] > 1e-10:
            fisher_mat += (np.outer(pi_d[:,i], pi_d[:,i]))/pi_s[i]
        else:
            continue

    covar       = np.linalg.inv(fisher_mat)
    covar_trace = np.trace(covar)

    return covar_trace



