import lin_space as lin
import exp_space as exp 
import numpy as np
import qutip as qt


np.set_printoptions(linewidth=np.inf)

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


def expect_given_D(d, deriv,n):
    """
    f: float, control pulse on X_1
    g: float, control pulse on Z_1
    l: float, magnetic field value
    n: int 

    returns: numpy float array, bloch vector of 1st physical qubit
    """
    part1 = sum( [d.full()[0][i]*deriv.full()[1][i+n+1] for i in range((n+1)) ] )
    part2 = sum( [np.conj(d.full()[0][i])*deriv.full()[1][i] for i in range((n+1)) ] )
    part3 = sum( [deriv.full()[0][i]*d.full()[1][i+n+1] for i in range((n+1)) ] )
    part4 = sum( [np.conj(deriv.full()[0][i])*d.full()[1][i] for i in range((n+1)) ] )

    e_x = np.real(-2*(part1 + part2 + part3 + part4) )
    e_y = np.imag(-2*(part1 + part2 + part3 + part4 ))
    e_z = 2*np.abs(sum( [(d.full()[1][i]*np.conj(deriv.full()[1][i]) + deriv.full()[1][i]*np.conj(d.full()[1][i]) ) for i in range((n+1)) ] ) )
    return np.array([e_x, e_y, e_z])


def fisher_info_matrix(f,g,l,n, delta=1e-7):
    d = D(f,g,l,n)
    ddel = D(f,g,l+delta,n)
    dderiv = (ddel-d)/delta
    
    d_actual_out = (dderiv)
    # d_actual_out = dderiv*d
    # print(d, dderiv, d_actual_out)
    # print(np.array(d_actual_out.full()))
    vec = expect_given_D(d, d_actual_out,n)
    print(vec)
    vec_test = expect_given_D(d, d_actual_out*d,n)
    print(f"test {np.inner(vec_test,vec_test)}")
    return np.inner(vec,vec)

def fisher_info(f, g, l, n, fun, delta = 1e-7):
    vec   = fun(f, g, l, n)
    del_vec =  fun(f, g, l+delta, n)

    deriv = (del_vec-vec)/delta
    print(deriv)
    return np.inner(deriv,deriv)

if __name__ == "__main__":
    f = 0
    g = 0
    l = 1
    n = 2
    time = 1
    delta = 1e-7
    #print('lin space qfi    : ', fisher_info(f,g,l,n, lin.exp) )
    print('exp space qfi     : ', fisher_info(f,g,l,n, exp.D) )
    print('exp  jw space qfi : ', fisher_info(f,g,l,n, lin.m_a1_check) )
    print('lin  jw space qfi : ', fisher_info(f,g,l,n, lin.expect) )
    print('lin  jw space qfi : ', fisher_info(f,g,l,n, lin.expect_stays_linear) )
    print('lin stay in D qfi  :', fisher_info_matrix(f,g,l,n))
    """
    for i in range(5):
        f = np.random.rand()#
        g = np.random.rand()

        print("----------------------------------------------")
        print(f"f value: {f}, g value: {g}")
        print("----------------------------------------------")
        print('exp space qfi     : ', fisher_info(f,g,l,n, exp.D) )
        print('exp  jw space qfi : ', fisher_info(f,g,l,n, lin.m_a1_check) )
        print('lin  jw space qfi : ', fisher_info(f,g,l,n, lin.expect) )
        print('lin  jw space qfi : ', fisher_info(f,g,l,n, lin.expect_stays_linear) )
        print('lin stay in D qfi  :', fisher_info_matrix(f,g,l,n))
        print("----------------------------------------------")
    """

