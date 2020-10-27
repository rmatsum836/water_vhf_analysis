import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima


chunk_length = 200
directory = 'water_form/2ns'

trj = md.load('2ns/water250_3obw-d3_100-2100ps_traject.xyz',
              top='water4vanHoveS/water4vanHove_DFTB/water4vanHove_DFTB_200-1000ps/water_250.pdb')
trj._unitcell_angles = np.zeros((trj.n_frames, 3))
trj._unitcell_lengths = np.zeros((trj.n_frames, 3))

trj._unitcell_angles[:] = [90, 90, 90]
trj._unitcell_lengths[:] = [1.9569, 1.9569, 1.9569]

trj.time = trj.time * 0.01

r, t, g_r_t = compute_van_hove(trj=trj,
                               chunk_length=chunk_length,
                               water=True)

dt = get_dt(trj)

# Save output to text files
np.savetxt('{}/d3_vhf.txt'.format(directory),
    g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
    dt,
    np.unique(np.round(np.diff(trj.time), 6))[0],
))
np.savetxt('{}/d3_r.txt'.format(directory), r, header='# Times, ps')
np.savetxt('{}/d3_t.txt'.format(directory), t, header='# Positions, nm')

fig, ax = plt.subplots()

for j in range(chunk_length):
    plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))


ax.set_ylim((0, 3))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('{}/d3_vhf.pdf'.format(directory))
