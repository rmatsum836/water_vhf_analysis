import os

import peakutils
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima
from scipy.signal import argrelextrema, find_peaks_cwt



g_r_t = 1 + np.loadtxt('VHF_1811pure.txt')
t = np.loadtxt('t_1811pure.txt')[:,0]
r = 0.1 * np.loadtxt('R_1811pure.txt')[0]
r_low = np.where(r > 0.2)[0][0]
r_high = np.where(r < 0.35)[0][-1]
r_range = r[r_low:r_high]

chunk_length = len(t)

fig, ax = plt.subplots(figsize=(10,10))

for j in range(chunk_length):
    if t[j] > 0.6 and t[j] < 1.0:
        plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))
       
        g_range = g_r_t[j][r_low:r_high] 
        r_max, g_r_max = find_local_maxima(r_range, g_range, r_guess=0.28)
        #local_max = find_peaks_cwt(g_r_t[0], np.arange(1,3), noise_perc=2)
        local_max = peakutils.indexes(g_r_t[j])
        plt.plot(r_max, g_r_max, 'k.')
        #plt.plot(r[local_max], g_r_t[j][local_max], 'k.')


ax.set_xlim((0, 1.0))
ax.set_ylim((0, 2.5))
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.legend(loc=2,
       prop={'size':6},
       bbox_to_anchor=(1.05,1),
       borderaxespad=0.0)
plt.tight_layout()
plt.savefig('van-hove-function_update.pdf')
