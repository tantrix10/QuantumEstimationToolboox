

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pylab import cm
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
#mpl.rcParams['font.family'] = 'Avenir'
mpl.rcParams['lines.markersize']=2
plt.rcParams['font.size'] = 9
plt.rcParams['lines.linewidth'] = 1

# width of a column in PRL/PRA (in inches)
width = 246/72.27
# height = width*2/(1+np.sqrt(5))
height = width*0.75
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family']='serif'
#mpl.rcParams['axes.linewidth'] = 1



x          = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
one_copy   = [0.8642188, 0.91084382, 0.9551543,  1.00755585, 1.06465341,  1.14397242,  1.25646548,  1.44642504, 1.76352121,  2.60051306]

two_copy   = [0.8642188, 0.91077073, 0.95678958, 1.0050153,  1.05438195,  1.11075009,  1.18513959,  1.222925,   1.45178332,  2.10334284]

three_copy = [0.8642188, 0.9003358,  0.95486846, 1.0049268,  1.04978361,  1.0812148,   1.15510696,  1.2201248,  1.4365739,   2.0982474 ]

hcrb       = [0.864219213955978,0.88108644649976,0.914072949771302,0.945945192285601,0.98782123120241,1.03674920187277,1.1079554199231,1.22519394881923,1.42481073711984,2.06354970086287]

qc_two     = [0.18879435,  0.254544, 0.31718218,  0.4537727,  0.52770914,  0.7262495,  0.89803107,  1.33527971,  1.70355303,  2.53352271]


fig = plt.figure(figsize=(width, height))

#fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, .81, .81])# Plot and show our data


ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
ax.xaxis.set_tick_params(which='major', size=5  , width=1  , direction='in', bottom='on')
ax.yaxis.set_tick_params(which='major', size=5  , width=1  , direction='in', left='on'  )
ax.yaxis.set_tick_params(which='minor', size=2.5, width=.51, direction='in', left='on'  )


ax.set_xlabel(r'Noise, $\gamma$')
ax.set_ylabel('CRB')

ax.set_xlim(0, 0.99)
ax.set_ylim(0, 2.7)


"""
  Solid lines: channel HCRB $\bar{C}^{\textrm{H}}\of{\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  }$ (optimized over initial state) and $k$-copy bounds $\tilde{C}^{(k)}\of{\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  }$ (optimized over measurements on multiple copies and computed for the HCRB-optimal initial state, see Fig.~\ref{fig:crbs-cd}).
    These solid lines correspond to the CQ strategy (e.g. left diagram in

"""
ax.tick_params(pad=10)
wid = 1
#ax.plot(x, one_copy  ,'r', linewidth=wid, label = r'$\tilde{C}_*^{(1)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$, One copy'  )
#ax.plot(x, two_copy  ,'y', linewidth=wid, label = r'$\tilde{C}_*^{(2)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$, Two copy'  )
#ax.plot(x, three_copy,'b', linewidth=wid, label = r'$\tilde{C}_*^{(3)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$, Three copy')
#ax.plot(x, hcrb      ,'g', linewidth=wid, label = r'$\bar{C}^{\textrm{H}}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}} })$, HCRB'    )

ax.plot(x, one_copy  ,'r', linewidth=wid, label = r'$\tilde{C}_*^{(1)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$'  )
ax.plot(x, two_copy  ,'y', linewidth=wid, label = r'$\tilde{C}_*^{(2)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$'  )
ax.plot(x, three_copy,'b', linewidth=wid, label = r'$\tilde{C}_*^{(3)}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}}  })$'  )
ax.plot(x, hcrb      ,'g', linewidth=wid, label = r'$\bar{C}^{\textrm{H}}({\mathcal{E}_{\varphi,\gamma}^{\otimes{2}} })$')



m_s = 4
ax.plot(x, one_copy  ,'ro', ms = m_s)
ax.plot(x, two_copy  ,'yo', ms = m_s)
ax.plot(x, three_copy,'bo', ms = m_s)
ax.plot(x, hcrb      ,'go', ms = m_s)


ax.plot(x, qc_two,'k--', dashes = (5,5) , label = r'$\bar{C}^{(2)}_{*\mathsf{QC}}({\mathcal{E}_{\varphi,\gamma} })$')
ax.plot(x, qc_two,'ko')

ax.legend(bbox_to_anchor=(.5, .98), loc=1, frameon=True, fontsize=6)




plt.savefig('unitary_optim.pdf', bbox_inches='tight')

#plt.show()
