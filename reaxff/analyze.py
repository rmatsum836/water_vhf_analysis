import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

summary = np.loadtxt(
    'data/summary.txt',
    skiprows=7,
    usecols=(0, 2, 4),
)

L = (summary[:, 2] ** (1/3)) * 0.1

stride = 1

trj = md.load('data/try_convert_to_.xtc', top='data/top.pdb', stride=stride)[:-1]

trj.time = summary[:, 1][::4] * 0.001

L = (summary[:, 2] ** (1/3)) * 0.1

trj._unitcell_angles = 90 * np.ones(shape=(trj.n_frames, 3))
trj._unitcell_lengths =  np.vstack([L[::4], L[::4], L[::4]]).T

trj_sample = trj[100000:]

#del trj

chunk_length = 200
directory = 'water_form'
r, t, g_r_t = compute_van_hove(trj=trj,
                               chunk_length=chunk_length,
                               water=True,
                               partial=False)

dt = get_dt(trj)

# Save output to text files
np.savetxt('{}/vhf.txt'.format(directory),
    g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
    dt,
    np.unique(np.round(np.diff(trj.time), 6))[0],
))
np.savetxt('{}/r.txt'.format(directory), r, header='# Positions')
np.savetxt('{}/t.txt'.format(directory), t/0.0005, header='# Time')

fig, ax = plt.subplots()

for j in range(chunk_length):
    if j % 200 == 0:
        plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))


ax.set_ylim((0, 3))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('{}/vhf.pdf'.format(directory))
