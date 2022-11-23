from setup import run
#import multiprocessing as m
from itertools import product
import numpy as np
#pool = m.Pool(m.cpu_count())

times = [10, 20, 30, 40, 50]
slots = [10, 20, 30, 40, 50, 60, 70]

out = np.array([0,0,0],dtype = object)

for x in list(product(times, slots)):
	out = np.vstack((out, run(4, 'XY', x[0], x[1])) )
	np.save('out_XY', out)