import matplotlib.pyplot as plt
import numpy as np
import os


cfi  = np.load(os.path.join('../results/results_one_copy', 'cfi_res.npy' ), allow_pickle = True)
hcrb = np.load(os.path.join('../results/results_one_copy', 'hcrb_res.npy'), allow_pickle = True)
cfi2  = np.load(os.path.join('../results/results_multi_copy', 'cfi_res.npy' ), allow_pickle = True)[0,:]
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

#print(cfi2)




plt.plot(x, cfi, label = 'CFI opt')
plt.plot(x, cfi2, label = 'CFI opt, 2 copies')
plt.plot(x, hcrb, label = 'corresponding HCRB')
plt.scatter(x, cfi)
plt.scatter(x, cfi2)
plt.scatter(x, hcrb)
plt.xlabel('noise, dephasing')
plt.ylabel('CRB')
plt.legend()
plt.show()
