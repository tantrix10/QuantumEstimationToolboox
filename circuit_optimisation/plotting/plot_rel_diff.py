import matplotlib.pyplot as plt
import numpy as np
import os


cfi  = np.load(os.path.join('../results/results_one_copy', 'cfi_res.npy' ), allow_pickle = True)
hcrb = np.load(os.path.join('../results/results_one_copy', 'hcrb_res.npy'), allow_pickle = True)
cfi2  = np.load(os.path.join('../results/results_multi_copy', 'cfi_res.npy' ), allow_pickle = True)[0,:]
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]



a = [1- cfi2[i]/cfi[i]  for i in range(len(cfi))]
b = [1- hcrb[i]/cfi[i]  for i in range(len(cfi))]

plt.plot(x, a, label = '2-copy vs 1-copy relative difference')
plt.plot(x, b, label = '1-copy vs HCRB relative difference')
#plt.plot(x, hcrb, label = 'corresponding HCRB')
plt.scatter(x, a)
plt.scatter(x, b)
#plt.scatter(x, hcrb)
plt.xlabel('noise, dephasing')
plt.ylabel('CRB')
plt.legend()
plt.show()
