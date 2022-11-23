import numpy as np


def genetic(par1, par2):
    extension_rate = 0.001    
    if par1.g == par2.g:
        ch_g = par1.g
        splice_point = np.random.randint(len(par1.sq))
        ch_sq        = np.append(par1.sq[:splice_point], par2.sq[splice_point:])
        ch_sq        = mutate(ch_sq)
        
        splice_point = np.random.randint(len(par1.tq))
        ch_tq        = np.append(par1.tq[:splice_point], par2.tq[splice_point:])
        ch_tq        = mutate(ch_tq)
        

        
        if np.random.random() < extension_rate/par1.g:
            ch_g += 1
            ch_sq.append([np.random.choice(2) for i in range(n*(n+1))])
            ch_tq.append([np.random.choice(2) for i in range(n*(n-1))])
        
        child = circuit(n, ch_g.g, du, init = [ch_sq, ch_tq])
        child = min([child, par1, par2], key = attrgetter('qfim'))
        ###################################################################################
        ###################################################################################
    else:
        par_big, par_small = max([par1, par2], key=attrgetter('g')), min([par1, par2], key=attrgetter('g'))
        
        splice_point_sq = np.random.randint(len(par_small.sq))
        splice_point_tq = np.random.randint(len(par_small.tq))
        ###################################################################################
                
        ch1_sq        = np.append(par1.sq[:splice_point], par2.sq[splice_point:len(par_small.sq)])
        ch1_sq        = mutate(ch1_sq)
        

        ch1_tq        = np.append(par1.tq[:splice_point], par2.tq[splice_point:len(par_small.sq)])
        ch1_tq        = mutate(ch1_tq)
        
        ###################################################################################
        ch2_sq        = np.append(par1.sq[:splice_point], par2.sq[splice_point:])
        ch2_sq        = mutate(ch2_sq)
        

        ch2_tq        = np.append(par1.tq[:splice_point], par2.tq[splice_point:])
        ch2_tq        = mutate(ch2_tq)        
        
        ###################################################################################
        # need to add possibility of extension g += 1 here too
        if np.random.random() < extension_rate/par1.g:
            par_small += 1
            ch1_sq.append([np.random.choice(2) for i in range(n*(n+1))])
            ch1_tq.append([np.random.choice(2) for i in range(n*(n-1))])
            
            par_big += 1
            ch2_sq.append([np.random.choice(2) for i in range(n*(n+1))])
            ch2_tq.append([np.random.choice(2) for i in range(n*(n-1))])            
            
        child1 = circuit(n, par_small.g, du, init = [ch1_sq, ch1_tq])
        child2 = circuit(n, par_big.g  , du, init = [ch2_sq, ch2_tq])
        child  = min([child1, child2, par1, par2], key=attrgetter('qfim'))
    
    
    return child

def mutate(gene, mut_rate):
    #mut_rate = 1/len(gene)
    for i in range(len(gene)):
        prob = np.random.random()
        if prob < mut_rate:
            gene[i] = xor(gene[i],1)
    return gene
