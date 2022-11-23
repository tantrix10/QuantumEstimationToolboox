 import scipy.io
import numpy as np
import scipy.io as sio
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# width of a column in PRL/PRA (in inches)
width = 246/72.27
height = width*2/(1+np.sqrt(5))
# height = width*0.75

# Choose a global style
matplotlib.style.use('seaborn-deep')
# matplotlib.style.use('seaborn-colorblind')
# matplotlib.style.use('classic')

# Enables the use of TeX's math mode for rendering the numbers in the formatter.
matplotlib.rcParams['text.usetex'] = True
# set the tex family to serif
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rcParams['font.family']='serif'
#turn on the minor ticks
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True
# let the ticks point inside
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

# matplotlib.rcParams['axes.titlepad'] = 20
matplotlib.rcParams['lines.markersize']=2
plt.rcParams['font.size'] = 9
plt.rcParams['lines.linewidth'] = 1


matlab_data1=sio.loadmat('hb_etaspan_n2to14_neta9.mat')
matlab_data2=sio.loadmat('hb_etaspan_n2to14_neta1.mat')
matlab_data3 = sio.loadmat('CRBs.mat')
matlab_data4 = sio.loadmat('CRBsGrid.mat')
#data preparation
etavec_narrow = np.array(matlab_data3['etavecnarrow']).flatten()
ccrbs_narrow = np.array(matlab_data3['CCRBs'])
scrbs_narrow = np.array(matlab_data3['QCRBs'])

ccrbs = np.array(matlab_data4['CCRBsGrid'])
scrbs = np.array(matlab_data4['QCRBsGrid'])
etavec = np.array(matlab_data4['etavec']).flatten()
hcrbs = np.concatenate((np.array(matlab_data1['HCRBsRankYalmip']), np.array(matlab_data2['HCRBsRankYalmip'])), axis=0)
# etavec = np.concatenate((np.array(matlab_data1['etavec']).flatten(), np.array(matlab_data2['etavec']).flatten()))

rel_sld_hol=np.empty_like(scrbs)
rel_hol_cla=np.empty_like(hcrbs)
for i in range(7):
    rel_sld_hol[:,i]=1-scrbs[:,i]/hcrbs[:,i]
    rel_hol_cla[:,i] = 1-hcrbs[:,i]/ccrbs[:,i]

markerslist_long=['o','v','s','>','P','^','D','<']

for i in range(7):
    plt.plot(etavec, hcrbs[:, i], marker=markerslist_long[i])

for i in range(7):
    plt.plot(etavec, rel_sld_hol[:, i], marker=markerslist_long[i])

for i in range(7):
    plt.plot(etavec, rel_hol_cla[:, i], marker=markerslist_long[i])

fig, ax = plt.subplots(2, 1, sharex='col',figsize=[width,height])
for i in range(7):

    ax[0].plot(etavec, rel_sld_hol[:, i], marker=markerslist_long[i])
    ax[1].plot(etavec, rel_hol_cla[:, i], label=r'$N=' + str(2*(i+1))+'$', marker=markerslist_long[i])
ax[1].set_xlim(0.09, 1.01)
ax[1].set_ylim(-0.002, 0.042)

# ax[0].set_yscale('log')
# locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=8)
# locmaj = matplotlib.ticker.LogLocator(base=10, numticks=8)
# ax[0].yaxis.set_major_locator(locmaj)
# ax[0].yaxis.set_minor_locator(locmin)
# ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# locmaj2 = matplotlib.ticker.MultipleLocator()
# ax[1].yaxis.set_major_locator(locmaj2)

# ax[1].legend(frameon=True, ncol=1, loc='upper right', prop={'size': 5})
# h, l = ax[1].get_legend_handles_labels()

# ax[0].text(0,-5.5, r'\textbf{(a)}')
ax[0].text(0.13,0.25, r'\textbf{(a)}')
ax[1].text(0.13,0.03, r'\textbf{(b)}')
ax[0].set_ylabel(r'$ 1 - C^\mathrm{S} / C^\mathrm{H}$')
ax[1].set_ylabel(r'$ 1 - C^\mathrm{H} / C^\mathrm{C}_{\phi}$')
ax[1].set_xlabel(r'$\eta$')
ax[0].tick_params(labelsize=7)
ax[1].tick_params(labelsize=7)
ax[1].legend(loc='upper right', prop={'size': 4.5})
fig.subplots_adjust(hspace=0)

# legend = fig.legend(loc='center right', prop={'size': 5})
# frame = legend.get_frame()
# frame.set_alpha(None)
# frame.set_edgecolor('black')

fig.align_labels()
fig.savefig("test.pdf",bbox_inches='tight')
# fig.savefig("test.pgf", bbox_inches='tight')

# for i in range(8):
#     if i!=2 and i!=6:
#         plt.plot(gammas, resultslist[i].reldif, label=str(
#         i+2)+' qubits', marker=markerslist[i])
# plt.yscale('log')
# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$1 - C^\mathrm{S} / C^\mathrm{H}$')
# plt.legend()
# # plt.savefig(dir_path+'test.png', bbox_inches='tight')

# for i in range(8):
#     if i != 2 and i != 6:
#         plt.plot(gammas, normD[i], label=str(
#             i+2)+' qubits', marker=markerslist[i])
# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$\left\Vert D \right\Vert$')
# plt.legend()

# for i in range(8):
#     if i != 2 and i != 6:
#         plt.plot(normD[i], resultslist[i].reldif, label=str(
#             i+2)+' qubits', marker=markerslist[i])
# plt.yscale('log')
# plt.ylabel(r'$1 - C^\mathrm{S} / C^\mathrm{H}$')
# plt.xlabel(r'$\left\Vert D \right\Vert$')
# plt.legend()


# for i in range(5):
#         plt.plot(gammas, lxz[i], label=str(
#             i+2)+' qubits', marker=markerslist[i])
# plt.legend()

# gs = gridspec.GridSpec(2, 3)
# ax0 = plt.subplot(gs[0, :])
# ax11 = plt.subplot(gs[1, 0])
# ax12 = plt.subplot(gs[1, 1])
# ax13 = plt.subplot(gs[1, 2])
# #ax2  = plt.subplot(gs[2,:])
# for i in range(8):
#     if i != 2 and i != 6:
#         ax0.plot(gammas, resultslist[i].reldif, label=str(i+2), marker=markerslist[i])
#         ax11.plot(gammas, lxy[i],marker=markerslist[i])
#         ax12.plot(gammas, lxz[i], marker=markerslist[i])
#         ax13.plot(gammas, lyz[i], marker=markerslist[i])
# #ax0.set_ylim((0,0.03))
# ax0.legend()
# ax0.set_yscale('log')
# ax0.set_xlabel(r'$\gamma$')
# ax0.set_ylabel(r'$ 1 - C^\mathrm{S} / C^\mathrm{H}$')
# ax11.set_ylabel(r'$D_{1,2}$')
# ax12.set_ylabel(r'$D_{1,3}$')
# ax13.set_ylabel(r'$D_{2,3}$')
# ax11.set_xlabel(r'$\gamma$')
# ax12.set_xlabel(r'$\gamma$')
# ax13.set_xlabel(r'$\gamma$')
# ax12.ticklabel_format(style='sci', scilimits=(0, 4), axis='y')
# #plt.savefig("test.png",bbox_inches='tight')

# plt.show()

