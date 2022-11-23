import qutip           as qt
import numpy           as np
import itertools       as it
import scipy           as sci
import multiprocessing as m
import holevo          as h
from holevo import naghol_spd
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

#np.set_printoptions(precision=np.inf,linewidth = np.inf, threshold= np.inf)


def gener(states, variences, mutatation_rate, n):
    cost                 = [1/variences[i] for i in range(len(variences))] #need to check for zeros
    cost_sum             = sum(cost)

    mating_pool_relative = [round(((cost[i])/cost_sum)*100) for i in range(len(variences))]
    mating_pool          = []
    new_states           = []
    for i in range(len(mating_pool_relative)):
        mating_pool = np.append(mating_pool, [i]*int(mating_pool_relative[i]) )
    for i in range(len(states)):
        state1       = states[int(np.random.choice(mating_pool))]
        state2       = states[int(np.random.choice(mating_pool))]
        splice_point = np.random.randint(2**n)
        n_state      = np.append(state1[:splice_point], state2[splice_point:])
        n_state      = mutate(n_state, mutatation_rate, n)
        try:
            n_state      = qt.Qobj(n_state).unit()
        except:
            n_state  = qt.rand_ket(2**n)
        n_state      = qt.Qobj(n_state, dims = [[2 for i in range(n)],[1 for j in range(n)]] )
        new_states.append(n_state)   #= np.append(new_states, n_state)
    return new_states



def mutate(state, mutatation_rate, n):
    possibilities = [1,2,3]#substitution, insertion, deletion
    for i in range(len(state)): #each element has the same chance of a mutation occuring
        chance = np.random.random()
        if chance < mutatation_rate:
            mut = np.random.choice(possibilities) #if  mutation, pick one the options
            if mut == 1:
                state[i] += 2*np.random.random()-1 + (2*np.random.random()-1)*1j
            elif mut == 2:
                ele = np.random.choice(2**n)
                state[ele] = state[i]
            elif mut == 3:
                state[i] = 0
    return state


def opt_state(gamma, itter):
    n     = 2
    alpha = [0,0,0]
    #itter            = 500
    nos              = 5
    #gamma           = 0
    #comp             = bloch_comp(n)
    states           = [qt.tensor([qt.rand_ket(2)]*n) for i in range(nos)] 
    best_state       = [states[0], 10000]
    #rho_t            = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
    #dhors            = [Drho(rho_t[i],alpha, gamma, states[i], n) for i in range(nos) ]
    pool             = m.Pool(processes=2)
    lim              = 9/(4*n*(n+2))
    for i in range(itter):
        #var = list(pool.starmap(naghol_spd, [[states[i], alpha, gamma, n] for i in range(nos) ] ))
        var = [naghol_spd(states[i], alpha, gamma, n) for i in range(nos) ]
        for j in range(nos):
            if var[j] == None or var[j] < lim or var[j] == np.inf:
                states[j] = best_state[0]
                var[j]    = best_state[1]
            elif var[j] < best_state[1]:
                best_state[0] = states[j]
                best_state[1] = var[j]

        states      = gener(states, var, 1/2**n, n)
        #print(states)
        states[-1]  = best_state[0]
        #rho_t       = [final_state(alpha, gamma, states[i], n) for i in range(nos)]
        #dhors       = [Drho(rho_t[i],alpha,gamma,  states[i], n) for i in range(nos) ]
        """
        name = str(n)+'_qubits_'+str(gamma)  #plt.show()
        with open(name, 'w') as f:
            print(best_state,  file=f)
        """
        #print(best_state)
    return best_state




