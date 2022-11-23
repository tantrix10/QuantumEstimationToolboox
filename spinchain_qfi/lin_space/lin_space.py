import numpy as np 
import qutip as qt
# import exp_space as e

x = qt.sigmax()
y = qt.sigmay()
z = qt.sigmaz()
ide = qt.identity(2)

def Mm(f, g, l, n):
    l = l
    g = g
    f = f
    xy = 1

    M = np.zeros((2*(n+1),2*(n+1)))
    M[0, 1] = M[1, 0] = M[n + 2, 0] = M[0, n + 2] =  (1/2)*f
    M[n + 1, 1] = M[1, n + 1] = M[n + 1, n + 2] = M[n + 2, n + 1] = -(1/2)*f
    M[1,1] = -g + -l
    M[n + 2, n + 2] = g + l
    #yes these two loops can be collapsed but for now during testing I keep them like this for clarity
    for i in range(2,n+1):
        M[i,i] += -l
        M[i, i-1] = M[i-1, i ] = xy

    for i in range(n+3, 2*n+2):
        M[i, i] += l
        M[i - 1, i] = M[i, i - 1] = -xy
    
    return M
    
def D_no_controls(l, n):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    #initialise the input alpha vector
    alpha = np.zeros((2*(n)),dtype = complex)
    alpha[0] = 1
    alpha[n] = 1
    #alpha = np.ones((2*(n)))
    alpha = qt.Qobj(alpha)

    #set up the grand-dynamical matrix
    M = np.zeros((2*(n),2*(n)))


    #yes these two loops can be collapsed but for now during testing I keep them like this for clarity
    for i in range(n):
        M[i,i] += -l
        if i != 0:
            M[i, i-1] = M[i-1, i ] = 1

    for i in range(n, 2*n):
        M[i, i] += l
        if i != n:  
            M[i - 1, i] = M[i, i - 1] = -1
    
    #print(M)
    
    #calculate the corresponding unitary
    #M = qt.Qobj(M)
    #Mu = qt.Qobj.expm(-1j*M)

    ass = [a(i, n) for i in range(n)]
    asd = [i.dag() for i in ass]
    ass.extend(asd)


    H = sum([M[j][i]*ass[j].dag()*ass[i] for i in range(2*(n) ) for j in range(2*(n))])
    #H = sum([ass[i].dag()*H[i] for i in range(2*(n+1)) ])
    print(H)

    u = qt.Qobj.expm(-1j*H)
    #H = ass.transpose() @ M @ ass
    #print('M mat ',H)
    #print(u)
    a0 = a(1,n)
    
    #print(u,a0)
    
    a0d = u.dag()*a0*u 
    #print('m test',a0d)
    #print(u*a(0,n+1)*u.dag())
    a0dd = u.dag()*a0.dag()*u 
    X = a0d + a0dd
    #print('JW exp space: ',X)
    Y = 1j*(a0dd-a0d)
    Z = a0d*a0dd - a0dd*a0d 
    #print(Z)
    zero     = qt.basis(2,0)
    in_state = qt.tensor([zero]*(n))
    in_state = in_state*in_state.dag()
    #print(in_state)
    e_x = np.real(qt.Qobj.tr(X*in_state))
    e_y = np.real(qt.Qobj.tr(Y*in_state))
    e_z = np.real(qt.Qobj.tr(Z*in_state))

    return np.array([e_x, e_y, e_z])

def D(f, g, l, n, t=1):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    n.b. input alpha vector for now is implicitly \alpha = (0,1, ..., 0_n+1, 0, 1, ..., 0_2n+2 )

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    
    M = Mm(f, g, l, n)
    #print('M: ',M)
    
    #calculate the corresponding unitary
    M = qt.Qobj(M)
    #print(M)
    Mu = qt.Qobj.expm(-1j*(t)*M)

    return Mu





def a(i,n):
    """
    returns a qutip quantum object representing the a_i annihilation operator
    i: int, the index of the site to be acted on
    n: int, number of qubits in system

    returns: qutip quantum object
    
    """
    a = (x+1j*y)/2
    # print(a, a.dag())
    # print(a.dag()*qt.basis(2,0))
    ids = qt.tensor([ide]*(n-i-1)) if n-i-1 > 0 else qt.Qobj(1)
    zs = qt.tensor([z]*i) if i > 0 else qt.Qobj(1)
    return qt.Qobj(qt.tensor([zs,a,ids]), dims = [[2]*n,[2]*n])


def a_dyn(U, n, j):
    """
    returns the evolved state of a_1 annihilation operator given a unitary matrix U and number of qubits n
    U: qutip quantum object, untiary evolution to act on operators a_i...a_i^d
    n: int, number of qubits

    returns: qutip quantum object, a_1
    """

    #a0 = a(0, n+1)
    #update = (a0.dag()-a0)
    #update2 = a0 - a0.dag()

    #ass = [update*a(i, n+1) if i != 0 else a0 for i in range(n+1)]
    #asd = [a(i, n+1).dag()*update2 if i != 0 else a0.dag() for i in range(n+1)]

    #ass = [update*a(i, n+1)  for i in range(n+1)]
    #asd = [a(i, n+1).dag()*update2 for i in range(n+1)]

    ass = [a(i, n+1)  for i in range(n+1)]
    asd = [i.dag() for i in ass]

    
    ass.extend(asd)

    #print(U)
    U = U.full()
    # U[j,0] = 0
    # U[j,n+2] = 0
    a1 = sum([U[j][i]*ass[i] for i in range(2*(n+1))])

    return a1

def expect(f,g,l,n,t=1):
    """
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    d = D(f, g, l, n, t) 
    #print(d)

    #a1 = ( a(0,n+1).dag() - a(0,n+1)  )*a1

    #a1 = ( a_dyn(d, n, 0).dag() - a_dyn(d, n, 0)  )*a1
    a0 = a(0,n+1)
    a0 = a_dyn(d, n, 0)
    update = (a0.dag()-a0)
    #print('d: ',d)
    a1 = a_dyn(d, n, 1)
    a1d = a1.dag()
    #print('lin d: ',d)
    #print(2*sum( [np.abs(d.full()[1][i])**2 for i in range(2*(n+1)) ] )-1  )
    #print('z test: ',2*sum( [np.abs(d.full()[1][i])**2 for i in range((n+1)) ] )-1  )
    #part1 = sum( [d.full()[0][i]*d.full()[1][i+n+1] for i in range((n+1)) ] )
    #part2 = sum( [np.conj(d.full()[0][i])*d.full()[1][i] for i in range((n+1)) ] )
    #print('x test: ', -2*(part1+part2)    )
    #print('a dyn: ', a_dyn(d, n, 1))
    # cor = [z] 
    # cor.extend([ide]*(n))   
    # cor = qt.tensor(cor)

    #sig =  cor*a1
    sig =  -update.dag()*a1
    #sig = a1
    

    #print(update, cor)

    sigd = sig.dag()
    X    = (sig + sigd)
    Y    = 1j*(sigd-sig)
    #print('lin space x: ',X)
    # X = a1 + a1.dag()
    # Y = 1j*(a1-a1.dag())
    Z = 2*a1*a1d - qt.tensor([ide]*((n+1)))
    #print(X,Y,Z)
    #print(a1.dag()-a1)
    #print(Z)
    #print(Z - Z.dag())
    plus = [(qt.basis(2,0) + qt.basis(2,1)  ).unit()]
    plus.extend([qt.basis(2,0)]*(n))
    in_state = qt.tensor(plus)
    #in_state = qt.tensor([qt.basis(2,0)]*(n+1))
    in_state = in_state*in_state.dag()
    #print(in_state)
    e_x = qt.Qobj.tr(X*in_state)
    e_y = qt.Qobj.tr(Y*in_state)
    e_z = qt.Qobj.tr(Z*in_state)

    return np.array([e_x, e_y, e_z])

def expect_stays_linear(f,g,l,n,t=1):
    """
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    d = D(f, g, l, n, t) 

    part1 = sum( [d.full()[0][i]*d.full()[1][i+n+1] for i in range((n+1)) ] )
    part2 = sum( [np.conj(d.full()[0][i])*d.full()[1][i] for i in range((n+1)) ] )
    e_x = np.real(-2*(part1+part2) )
    e_y = np.imag(-2*(part1+ part2 ))
    #d_fac = sum(np.abs(d.full()[0][i])**2 for i in range(n))
    #d_fac = np.real(sum(d.full()[0]))
    #print(d_fac)
    #e_z =d_fac*(2*sum( [np.abs(d.full()[1][i])**2 for i in range((n+1)) ] )-1 )
    e_z = 2*sum( [np.abs(d.full()[1][i]   )**2 for i in range((n+1)) ]  ) -1
    return np.array([e_x, e_y, e_z])

def expect_given_D(d):
    """
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    # THIS NEEDS TO BE REMOVED AT SOME POINT
    n = 2
    part1 = sum( [d.full()[0][i]*d.full()[1][i+n+1] for i in range((n+1)) ] )
    part2 = sum( [np.conj(d.full()[0][i])*d.full()[1][i] for i in range((n+1)) ] )
    e_x = np.real(-2*(part1+part2) )
    e_y = np.imag(-2*(part1+ part2 ))
    e_z = 2*sum( [np.abs(d.full()[1][i])**2 for i in range((n+1)) ] )-1 
    #d_fac = np.real(sum(d.full()[0]))
    #print(d_fac)
    #e_z =d_fac*(2*sum( [np.abs(d.full()[1][i])**2 for i in range((n+1)) ] )-1 )
    #
    return np.array([e_x, e_y, e_z])


def jw_a1_check(f,g,l,n):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    n.b. input alpha vector for now is implicitly \alpha = (0,1, ..., 0_n+1, 0, 1, ..., 0_2n+2 )

    returns: qutip quantum object, hamiltonian in JW-picture in terms of creation and anihilation operators
    


    """
    a0 = a(0,n)
    u = e.U(f,g,l,n)
    a0d = u.dag()*a0*u 
    #print(a0d)
    a0dd = u.dag()*a0.dag()*u 
    X = a0d + a0dd
    #print('JW exp space: ',X)
    Y = 1j*(a0dd-a0d)
    Z = a0d*a0dd - a0dd*a0d 
    zero     = qt.basis(2,0)
    # plus     =  (qt.basis(2,0)+qt.basis(2,1)).unit()
    # states   = [plus]
    # states.extend([zero]*(n-1))
    # in_state = qt.tensor(states)
    in_state = qt.tensor([zero]*n)
    in_state = in_state*in_state.dag()

    

    e_x = np.real(qt.Qobj.tr(X*in_state))
    e_y = np.real(qt.Qobj.tr(Y*in_state))
    e_z = np.real(qt.Qobj.tr(Z*in_state))

    return np.array([e_x, e_y, e_z])


def m_a1_check(f,g,l,n):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    n.b. input alpha vector for now is implicitly \alpha = (0,1, ..., 0_n+1, 0, 1, ..., 0_2n+2 )

    returns: qutip quantum object, hamiltonian in JW-picture in terms of creation and anihilation operators
    
    """
    ass = [a(i, n+1) for i in range(n+1)]
    asd = [i.dag() for i in ass]
    ass.extend(asd)
    #ass = np.array(ass,)
    M = Mm(f, g, l, n)

    H = sum([M[j][i]*ass[j].dag()*ass[i] for i in range(2*(n+1) ) for j in range(2*(n+1))])
    #H = sum([ass[i].dag()*H[i] for i in range(2*(n+1)) ])
    #print(H)
    u = qt.Qobj.expm(-1j*(1/2)*H)
    #print('U: ',u)
    
    #H = ass.transpose() @ M @ ass
    #print('M mat ',H)
    #print(u)
    a0 = a(1,n+1)
    ac = a(0,n+1)
    a0 = (ac.dag()-ac)*a0
    #print(u,a0)
    #print(u.dag()*a0*u)
    a0d = u.dag()*a0*u 
    #print('m test',a0d)
    #print('m test: ', u.dag()*a0*u)
    a0dd = u.dag()*a0.dag()*u 
    X = a0d + a0dd
    #print('JW exp space: ',X)
    Y = 1j*(a0dd-a0d)
    Z = a0d*a0dd - a0dd*a0d 
    #print(Z)
    # zero     = qt.basis(2,0)
    # in_state = qt.tensor([zero]*(n+1))
    # in_state = in_state*in_state.dag()
    #print(in_state)

    plus = [(qt.basis(2,0) + qt.basis(2,1)  ).unit()]
    plus.extend([qt.basis(2,0)]*(n))
    in_state = qt.tensor(plus)
    #in_state = qt.tensor([qt.basis(2,0)]*(n+1))
    in_state = in_state*in_state.dag()

    e_x = np.real(qt.Qobj.tr(X*in_state))
    e_y = np.real(qt.Qobj.tr(Y*in_state))
    e_z = np.real(qt.Qobj.tr(Z*in_state))

    return np.array([e_x, e_y, e_z])



def pure_a_a1_check(f,g,l,n):
    """
    function to generate the dynamical matrix given two constant control pulses, f & G, a constant magnetic field l and a number of sites n
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    n.b. input alpha vector for now is implicitly \alpha = (0,1, ..., 0_n+1, 0, 1, ..., 0_2n+2 )

    returns: qutip quantum object, hamiltonian in JW-picture in terms of creation and anihilation operators
    


    """
    ass = [a(i, n) for i in range(n)]
    asd = [i.dag() for i in ass]
    #ass.extend(asd)
    #ass = np.array(ass,)

    hinter = sum([asd[i]*ass[i+1] + asd[i+1]*ass[i]  for i in range(n-1)])
    #print([asd[i]*ass[i+1] + asd[i+1]*ass[i]  for i in range(n-1)])
    
    hz     = sum([-asd[i]*ass[i] + ass[i]*asd[i] for i in range(n)])

    hx  = ass[0]+ asd[0]
    hzc = -asd[0]*ass[0] + ass[0]*asd[0]
    #print(hinter, hz, hx, hzc)
    #print(hinter)
    H = 2*hinter + l*hz + f*hx + g*hzc

    u = qt.Qobj.expm(-1j*(1/2)*H)
    #H = ass.transpose() @ M @ ass
    #print('JW Hamiltonian',H)
    a0 = a(0,n)
    
    #print(u.dag()*a0*u)
    
    a0d = u.dag()*a0*u 
    #print(a0d)

    a0dd = u.dag()*a0.dag()*u 
    #print('a: ',a0dd)
    X = a0d + a0dd
    #print('JW exp space: ',X)
    Y = 1j*(a0dd-a0d)
    Z = a0d*a0dd - a0dd*a0d 
    zero     = qt.basis(2,0)
    in_state = qt.tensor([zero]*(n))
    in_state = in_state*in_state.dag()
    e_x = np.real(qt.Qobj.tr(X*in_state))
    e_y = np.real(qt.Qobj.tr(Y*in_state))
    e_z = np.real(qt.Qobj.tr(Z*in_state))

    return np.array([e_x, e_y, e_z])


def com_mat(n):
    ass = [a(i, n) for i in range(n)]
    asd = [i.dag() for i in ass]
    ass.extend(asd)

    mat = np.zeros((2*n,2*n))

    for i, a1 in enumerate(ass):
        for j, a2 in enumerate(ass):
            mat[i,j] = 1 if np.count_nonzero( (a1*a2 + a2*a1).full() ) != 0 else 0
    return qt.Qobj(mat)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
    f = 2
    g = 3
    l = 1
    n = 2
    print(n)#
    #print([a(i, 3) for i in range(3)])
    
    #print("lin space, no controls: ",D_no_controls(l, n))
    print("exp space, controls   : ",e.D(f,g,l,n))
    print("exp space pure jw crls: ",pure_a_a1_check(f,g,l,n))
    print("exp space m, controls : ",m_a1_check(f,g,l,n))
    print("lin space, controls   : ",expect(f,g,l,n,t=1))
    print("pure lin,  controls   : ",expect_stays_linear(f,g,l,n,t=1))
    from functools import reduce
    fc = np.random.random(10)
    fi,gi = np.array_split(fc, 2)
    import time 

    a = time.time()
    unis = [D(fi[i], gi[i], l, n) for i in range(len(fi))]
    final = expect_given_D(reduce(lambda x,y: x*y, unis))
    print('lin space time check: ',final)
    b = time.time()
    

    a2 = time.time()
    unis = [e.U(fi[i], gi[i], l, n) for i in range(len(fi))]
    final = e.D_given_u(reduce(lambda x,y: x*y, unis),n)
    print('exp space time check: ',final)
    b2 = time.time()

    print('linear time: ', b-a)
    print('exp time: ',b2-a2)




    ###################################################
    #print(com_mat(3))
    #print(D_no_controls(l, n))
    #print(Mm(f, g, l, n))


    # zero     = qt.basis(2,0)
    # in_state = qt.tensor([zero]*(n))
    # in_state = in_state*in_state.dag()
    # e_x = np.real(qt.Qobj.tr(a(1,n)*a(1,n).dag()*in_state))
    # print(e_x)

    f = 2
    g = 3
    l = 1
    n = 2
    delta = 1e-7
    time = 1
    d1 = D(f,g,l,n,t=time)
    print("D1: ", d1)
    d2 = D(f,g,l+delta,n,t=time)
    deriv = (d2-d1)/delta
    print("DERIV:", deriv)
    print("Expectations D, Deriv: ",expect_given_D(deriv) )
    e1 = expect_given_D(d1)
    e2 = expect_given_D(d2)
    print("expec D, functional : ", (e2-e1)/delta)





