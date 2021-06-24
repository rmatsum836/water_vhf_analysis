import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima
from scattering.utils.run import run_total_vhf


chunk_length = 200
n_chunks = 3000
directory = 'overlap_nvt'

trj = md.load('sample_nvt_2.xtc',
              top='sample_nvt_2.gro')
dt = get_dt(trj)

r, t, g_r_t = run_total_vhf(trj, chunk_length, n_chunks)

np.savetxt('{}/vhf_final.txt'.format(directory),
    g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
    dt,
    np.unique(np.round(np.diff(trj.time), 6))[0]))
np.savetxt('{}/r_final.txt'.format(directory), r, header='# Positions')
np.savetxt('{}/t_final.txt'.format(directory), t, header='# Time')

fig, ax = plt.subplots()

for j in range(chunk_length):
    if j % 10 == 0:
        plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))

ax.set_ylim((0, 3))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('{}/vhf_final.pdf'.format(directory))
