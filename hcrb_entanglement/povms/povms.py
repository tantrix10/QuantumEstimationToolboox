#import circuit
#import cvxpy     as cp
import qutip as qt
import scipy     as sci
import numpy as np
import numpy.matlib
import cvxpy as cp
#


sx     = qt.sigmax()
sy     = qt.sigmay()
sz     = qt.sigmaz()
si     = qt.identity(2)
v1, d1 = qt.Qobj.eigenstates(sx)
v2, d2 = qt.Qobj.eigenstates(sy)
v3, d3 = qt.Qobj.eigenstates(sz)




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



def hcrb(rho, drho, phi_0):
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
    prob.solve(solver = cp.CVXOPT)#, verbose = True)
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
    #print(fisher_mat)
    try:
        covar       = np.linalg.inv(fisher_mat)
        covar_trace = np.trace(covar)
    except:
        covar_trace = 1e5
    if covar_trace < 0:
        covar_trace = 1e5
    return covar_trace


def sld_prod(rho, x1, x2):
    return qt.Qobj.tr(rho*(x1*x2+x2*x1)/2)


def deriv(rho):
    return [-1j*(du[q]*rho-rho*du[q].dag())  for q in range(3)]



def hamiltonian(pauli, n):
        return sum( [qt.tensor([si if i !=j else pauli for i in range(n) ]) for j in range(n)])


def sld(rho):
    temp = deriv(rho)
    return [2*temp[i] for i in range(len(temp))]
    


def qfim(rho):
    L = sld(rho)
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(i+1):
            mat[i,j] = mat[j,i] = sld_prod(rho, L[i], L[j])

    a = np.linalg.inv(mat)
    return a, mat


def weird_prod(rho, dual, a):
    return sum([sld_prod(rho, dual[i],a)**2 for i in range(3)])


def opt_check(x0):
    rho   = phi*phi.dag()
    drho  = deriv(rho)
    ch    = hcrb(rho, drho, phi)
    jsin, js  = qfim(rho)
    slds  = sld(rho)
    d, v  = np.linalg.eig(jsin)
    dual  = [sum([jsin[i,j]*slds[j] 
    			for j in range(3)]) 
    				for i in range(3)]
    dual = slds[:]
    A     = [sum([v[i,j]*dual[j] for j in range(3)]) for i in range(3)]
    A.append(rho)
    Avecs = [qt.Qobj.eigenstates(A[i]) for i in range(len(A))]
    Eops  = [[Avecs[i][1][j]*Avecs[i][1][j].dag() 
    			for j in range(len(Avecs[i][1])) ] 
    					for i in range(len(A))]

    x0 = np.abs(x0)
    xsum = sum(x0)
    x = [i/xsum for i in x0]
    povms = [(x[0]) * Eops[0][0], (x[0]) * Eops[0][1], (x[0]) * Eops[0][2], (x[0]) * Eops[0][3], 

             (x[1]) * Eops[1][0], (x[1]) * Eops[1][1], (x[1]) * Eops[1][2], (x[1]) * Eops[1][3],

             (x[2]) * Eops[2][0], (x[2]) * Eops[2][1], (x[2]) * Eops[2][2], (x[2]) * Eops[2][3],

             (x[3]) * Eops[3][0], (x[3]) * Eops[3][1], (x[3]) * Eops[3][2], (x[3]) * Eops[3][3]]

    return fisher_info(rho, drho, povms)


def estim(Avecs, rho, A, povms, dual, p1, p2, drho):
    est11 = [[np.sqrt(2)   * Avecs[2][0][i] *sld_prod(rho,dual[j],A[2]) for i in range(4)] for j in range(3)]
    est21 = [[np.sqrt(2)/p2* Avecs[0][0][i] *sld_prod(rho,dual[j],A[0]) for i in range(4)] for j in range(3)]
    est31 = [[np.sqrt(2)/p1* Avecs[1][0][i] *sld_prod(rho,dual[j],A[1]) for i in range(4)] for j in range(3)]

    var11 = sum([sum([(est11[j][i]**2) for j in range(3)]) * qt.Qobj.tr(rho*povms[i+8]) for i in range(4)] )
    var21 = sum([sum([(est21[j][i]**2) for j in range(3)]) * qt.Qobj.tr(rho*povms[ i ]) for i in range(4)] )
    var31 = sum([sum([(est31[j][i]**2) for j in range(3)]) * qt.Qobj.tr(rho*povms[i+4]) for i in range(4)] )
    var1sum = var31+var21+var11


    loc1 = sum([sum([(est11[0][i]) for j in range(3)]) * qt.Qobj.tr(rho*povms[i+8]) for i in range(4)] )
    loc2 = sum([sum([(est21[0][i]) for j in range(3)]) * qt.Qobj.tr(rho*povms[ i ]) for i in range(4)] )
    loc3 = sum([sum([(est31[0][i]) for j in range(3)]) * qt.Qobj.tr(rho*povms[i+4]) for i in range(4)] )
    print('local condition: ', f'{loc1:.4}',f'{loc2:.4}',f'{loc3:.4}')


    unbiased = np.zeros((3,3));
    uni1 = [sum([sum([(est11[0][i]) for j in range(3)]) * qt.Qobj.tr(drho[k]*povms[i+8]) for i in range(4)] ) for k in range(3)]
    uni2 = [sum([sum([(est21[0][i]) for j in range(3)]) * qt.Qobj.tr(drho[k]*povms[ i ]) for i in range(4)] ) for k in range(3)]
    uni3 = [sum([sum([(est31[0][i]) for j in range(3)]) * qt.Qobj.tr(drho[k]*povms[i+4]) for i in range(4)] ) for k in range(3)]

    unbiased[0,:] = np.real(uni1)
    unbiased[1,:] = np.real(uni2)
    unbiased[2,:] = np.real(uni3)

    print(unbiased)
 
    return var1sum


def estim_simple(Avecs, rho, A, povms, dual, p1, p2, drho):

    a = 1/0.0488281 #1/0.44194174
    b = 1/0.0046921#np.sqrt(2)/(p2*0.44194174)
    c = 1/0.27027027#np.sqrt(2)/(p1*0.44194174)

    print(1/a+1/b+1/c)

    est11 = [a * Avecs[2][0][i] *sld_prod(rho,dual[1],A[2]) for i in range(4)];
    est21 = [b * Avecs[0][0][i] *sld_prod(rho,dual[1],A[0]) for i in range(4)];
    est31 = [c * Avecs[1][0][i] *sld_prod(rho,dual[1],A[1]) for i in range(4)];


    est12 = [a * Avecs[2][0][i] *sld_prod(rho,dual[2],A[2]) for i in range(4)];
    est22 = [b * Avecs[0][0][i] *sld_prod(rho,dual[2],A[0]) for i in range(4)];
    est32 = [c * Avecs[1][0][i] *sld_prod(rho,dual[2],A[1]) for i in range(4)];

    est10 = [a * Avecs[2][0][i] *sld_prod(rho,dual[0],A[2]) for i in range(4)];
    est20 = [b * Avecs[0][0][i] *sld_prod(rho,dual[0],A[0]) for i in range(4)];
    est30 = [c * Avecs[1][0][i] *sld_prod(rho,dual[0],A[1]) for i in range(4)];



    var11 = sum([(est11[i]**2)*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var21 = sum([(est21[i]**2)*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var31 = sum([(est31[i]**2)*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])



    var12 = sum([(est12[i]**2)*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var22 = sum([(est22[i]**2)*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var32 = sum([(est32[i]**2)*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])




    var10 = sum([(est10[i]**2)*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var20 = sum([(est20[i]**2)*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var30 = sum([(est30[i]**2)*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])

    var1sum = var31 + var21 + var11 
    var2sum = var32 + var22 + var12
    var0sum = var30 + var20 + var10 

##############################################################################

    var11 = sum([(est10[i])*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var21 = sum([(est20[i])*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var31 = sum([(est30[i])*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])
    estim1 = var31+var21+var11 

    var12 = sum([(est11[i])*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var22 = sum([(est21[i])*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var32 = sum([(est31[i])*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])
    estim2 = var32+var22+var12


    var10 = sum([(est12[i])*qt.Qobj.tr(rho*povms[i+8]) for i in range(4)])
    var20 = sum([(est22[i])*qt.Qobj.tr(rho*povms[i]  ) for i in range(4)])
    var30 = sum([(est32[i])*qt.Qobj.tr(rho*povms[i+4]) for i in range(4)])
    estim3 = var30+var20+var10 
    print('local condition: ', np.isclose(estim1,0), np.isclose(estim2,0), np.isclose(estim3,0))
#############################################################################
    unbiased = np.zeros((3,3))

    var11 = [sum([(est10[i])*qt.Qobj.tr(drho[k]*povms[i+8]) for i in range(4)]) for k in range(3)]
    var21 = [sum([(est20[i])*qt.Qobj.tr(drho[k]*povms[i]  ) for i in range(4)]) for k in range(3)]
    var31 = [sum([(est30[i])*qt.Qobj.tr(drho[k]*povms[i+4]) for i in range(4)]) for k in range(3)]
    estim1 = np.array(var31)+np.array(var21)+np.array(var11) 

    var12 = [sum([(est11[i])*qt.Qobj.tr(drho[k]*povms[i+8]) for i in range(4)]) for k in range(3)]
    var22 = [sum([(est21[i])*qt.Qobj.tr(drho[k]*povms[i]  ) for i in range(4)]) for k in range(3)]
    var32 = [sum([(est31[i])*qt.Qobj.tr(drho[k]*povms[i+4]) for i in range(4)]) for k in range(3)]
    estim2 = np.array(var32)+np.array(var22)+np.array(var12)


    var10 = [sum([(est12[i])*qt.Qobj.tr(drho[k]*povms[i+8]) for i in range(4)]) for k in range(3)]
    var20 = [sum([(est22[i])*qt.Qobj.tr(drho[k]*povms[i]) for i in range(4)]) for k in range(3)]
    var30 = [sum([(est32[i])*qt.Qobj.tr(drho[k]*povms[i+4]) for i in range(4)]) for k in range(3)]
    estim3 = np.array(var30)+np.array(var20)+np.array(var10)
    #print(estim1, estim2, estim3)
    unbiased[0,:] = np.real(estim1)
    unbiased[1,:] = np.real(estim2)
    unbiased[2,:] = np.real(estim3)

    print(unbiased)



    return var0sum+var1sum+var2sum


def sld_test(slds, dual, rho):
    out = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            out[i,j] = sld_prod(rho, slds[i], dual[j])
    return out

def check(phi):
    rho  = phi*phi.dag()
    drho = deriv(rho)
    ch   = hcrb(rho, drho, phi)
    jsin, js = qfim(rho)
    slds = sld(rho)
    d, v = np.linalg.eig(jsin)
    dual = [sum([jsin[i,j]*slds[j] for j in range(3)]) for i in range(3)]
    dual = slds[:]
    A    = [sum([v[i,j]*dual[j] for j in range(3)]) for i in range(3)]
    Avecs = [qt.Qobj.eigenstates(A[i]) for i in range(3)]
    #print(Avecs)
    Eops  = [[Avecs[i][1][j]*Avecs[i][1][j].dag() for j in range(len(Avecs[i][1])) ] for i in range(3)]
    u1 = weird_prod(rho, dual, A[1])
    u2 = weird_prod(rho, dual, A[2])
    p1 = np.sqrt(u1)/(np.sqrt(u1)+np.sqrt(u2))
    p2 = np.sqrt(u2)/(np.sqrt(u1)+np.sqrt(u2))
    #print(Eops[1][1], p1,p2)
    povms = [(0.5*p2) * Eops[0][0], (0.5*p2) * Eops[0][1], (0.5*p2) * Eops[0][2], (0.5*p2) * Eops[0][3],

             (0.5*p1) * Eops[1][0], (0.5*p1) * Eops[1][1], (0.5*p1) * Eops[1][2], (0.5*p1) * Eops[1][3],

             (0.5)    * Eops[2][0], (0.5)    * Eops[2][1], (0.5)    * Eops[2][2], (0.5)    * Eops[2][3]]
    np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
    #print(povms)
    povms = [(1/3)*Eops[0][0],(1/3)*Eops[0][1],(1/3)*Eops[0][2],(1/3)*Eops[0][3], 
       (1/3)*Eops[1][0],(1/3)*Eops[1][1],(1/3)*Eops[1][2],(1/3)*Eops[1][3],
       (1/3)*Eops[2][0],(1/3)*Eops[2][1],(1/3)*Eops[2][2],(1/3)*Eops[2][3]]
    #print(d)
    eigch = d[0]+d[1]+d[2]+2*np.sqrt(d[1]*d[0])
    print('hcrb: 'f'{ch:.4}', ',  eig: ', f'{eigch:.4}','\n', 'fisher: '  f'{fisher_info(rho, drho, povms):.4}', sep='')
    #print( f'{eigch:.4}','\n', f'{fisher_info(rho, drho, povms):.4}', sep='')
    #print('var :',estim(Avecs, rho, A, povms,dual, p1, p2, drho))
    #print('var simple: ',estim_simple(Avecs, rho, A, povms, dual, p1, p2, drho))
    #print('sld test: \n',sld_test(slds, dual, rho))




du  = [hamiltonian(sx, 2), hamiltonian(sy, 2), hamiltonian(sz, 2)]

#phi = qt.Qobj(qt.rand_ket(4), dims = [[2,2],[1,1]]).unit()
phi = qt.Qobj([[np.sqrt(0.8)],[0],[0],[np.sqrt(0.2)]], dims = [[2,2],[1,1]]).unit()
#check(phi)
sol = sci.optimize.minimize(opt_check, np.random.random(4))
print('optimised fisher: ',sol.fun, ' ', sol.message)
check( phi)


