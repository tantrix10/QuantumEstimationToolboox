import numpy as np
import matplotlib.pyplot as plt


def hcrb(r1,r2,r3):
	"""takes the three elements of a perm invar real valued state
	and return HCRB for 3D mag field with zero para
	"""

	p1  = 4*np.sqrt( 1/ (  (  (r1+r3)**2)*((r1-r3 )**2 +4*r2**2)    ))  
	p2  = 2/((r1-r3)**2 + 4*r2**2)
	p3  = 2/(r1+r3)**2
	p41 = r1**6 + (r1**4)*(6*r2**2 -r3**2 - 2 ) + 8*(r1**3)*(r2**2)*r3
	p42 = (r1**2)*(8*r2**4 + 4*r2**2 * (r3**2 -2 ) -r3**4 + 4*r3**2 + 1  )
	p43 = 8 * r1 * (r2**2) * r3 *(2*r2**2 + r3**2 -2 )
	p44 = 2*r2**2 + (r3**2)*(8*r2**4 + (r2**2)*(6*r3**2 -8)+(r3**2-1)**2 )
	p4 = p41 + p42 + p43 + p44
	out = p1 + p2 + p3 + 1/p4 

	return (1/16)*(out), abs(r1*r3-r2**2)


def qfi(r1,r2,r3):
	p1 = 2/(4*r2**2 + (r1-r3)**2)
	p2 = 2/(r1+r3)**2
	p3 =  1/(-r1**4 - 8*r1*(r2**2)*r3 + r3**2 - r3**4 + (r2**2)*(2 - 4*r3**2) + (r1**2)*(1 - 4*r2**2 + 2*r3**2))
	return (1/16)*(p1+p2+p3)



den  = 50
vec  = np.linspace(-1,1,den)
res1 = []
res2 = []
res3 = []
for i in range(den):
	for j in range(den):
		for k in range(den):
			r1,r2,r3 = vec[i],vec[j],vec[k]
			b = np.sqrt(r1**2 + r2**2 + r2**2 + r3**2)
			#print(b)
			if b > 1e-10:
				a = 1/b
				#print(r1,r2,r3)
				r1, r2, r3 = r1*a, r2*a, r3*a
				#print(r1,r2,r3)
			else:
				continue
			try:
				hb, ent = hcrb(r1,r2,r3)
				qfi1     = qfi(r1,r2,r3)
				#print(hb,ent)
			except:
				continue
			if hb != "nan":
				#print(hb, ent)
				res1.append(hb)
				res2.append(ent)
				res3.append(qfi1)

import scipy.io
import numpy as np
import scipy.io as sio
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
import os


# width of a column in PRL/PRA (in inches)
width = 100
height = width*2/(1+np.sqrt(5))
# height = width*0.75

# Choose a global style
matplotlib.style.use('seaborn-deep')
# matplotlib.style.use('seaborn-colorblind')
# matplotlib.style.use('classic')

# Enables the use of TeX's math mode for rendering the numbers in the formatter.
#matplotlib.rcParams['text.usetex'] = True
# set the tex family to serif
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rcParams['font.family']='serif'
#turn on the minor ticks
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True
# let the ticks point inside
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

matplotlib.rcParams['lines.markersize']=2
plt.rcParams['font.size'] = 15
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
fig, ax = plt.subplots(1, 1, sharex='col',figsize=[width,height])

ax.scatter(res2, res3, label = 'QFI')
ax.scatter(res2, res1, label = 'HCRB')
ax.set_ylim((0,5))
ax.set_ylabel('CRB')
ax.set_xlabel('Pure concurrence '+  r'$|r_1r_2-r_3r_4|$')
ax.tick_params(labelsize=7)

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)


ax.legend(loc='upper right', prop={'size': 15.5})
fig.subplots_adjust(hspace=0)

ax.text(0.3,2.25, '(QFI optimal)')
ax.text(0.2,2.25, '(HCRB optimal)')

a1 = min(i for i in res3 if i>0)
a2 = min(i for i in res1 if i>0)

x1 = res3.index(a1)
x2 = res1.index(a2)

ax.axvline(x = res2[x1])
ax.axvline(x = res2[x2])

fig.align_labels()
plt.show()
fig.savefig("test.pdf",bbox_inches='tight')

#print(min(res3))
#print(max(res2), res1[np.argmax(res2)])
"""
plt.scatter(res2, res3, label = 'QFI')
plt.scatter(res2, res1, label = 'HCRB')
plt.ylim((0,5))
plt.xlabel('level of entanglement')
plt.ylabel('CRB')
plt.legend()
plt.show()
"""









