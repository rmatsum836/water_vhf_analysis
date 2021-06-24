import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima
from scattering.utils.run import run_total_vhf

chunk_length = 200
n_chunks = 15000
directory = 'overlap_nvt'

# Load trj
trj = md.load('2ns/water250_3obw-d3_100-2100ps_traject.xyz',
              top='water4vanHoveS/water4vanHove_DFTB/water4vanHove_DFTB_200-1000ps/water_250.pdb')
trj._unitcell_angles = np.zeros((trj.n_frames, 3))
trj._unitcell_lengths = np.zeros((trj.n_frames, 3))

trj._unitcell_angles[:] = [90, 90, 90]
trj._unitcell_lengths[:] = [1.9569, 1.9569, 1.9569]

trj.time = trj.time * 0.01
import pdb; pdb.set_trace()

dt = get_dt(trj)

print("Running vhf")
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
