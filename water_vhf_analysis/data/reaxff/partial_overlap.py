import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_partial_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima
from scattering.utils.run import run_partial_vhf

chunk_length = 100
n_chunks = 9000
#directory = 'partial_overlap_nvt'
directory = '9000_partial_overlap_nvt'

combos = [['O', 'O'],
          ['O', 'H'],
          ['H', 'H']]

summary = np.loadtxt(
    'data/summary.txt',
    skiprows=7,
    usecols=(0, 2, 4),
)

L = (summary[:, 2] ** (1/3)) * 0.1

stride = 1

# Cutting off first 200 ps of trajectory
trj = md.load('nvt_data/try_convert_to_.xtc', top='nvt_data/top.pdb', stride=stride)[10000:-1]
trj.time = summary[:, 1][::8][10000:] * 0.001

L = (summary[:, 2] ** (1/3)) * 0.1

trj._unitcell_angles = 90 * np.ones(shape=(trj.n_frames, 3))
trj._unitcell_lengths =  np.vstack([L[::8], L[::8], L[::8]]).T
dt = get_dt(trj)

for combo in combos:
    r, t, g_r_t = run_partial_vhf(trj, selection1=combo[0], selection2=combo[1], 
                                  chunk_length=chunk_length,
                                  self_correlation=False,
                                  n_chunks=n_chunks, water=True)

    np.savetxt('{}/vhf_final_{}_{}.txt'.format(directory,combo[0],combo[1]),
        g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
        dt,
        np.unique(np.round(np.diff(trj.time), 6))[0],
    ))

    np.savetxt('{}/r_final_{}_{}.txt'.format(directory,combo[0],combo[1]),
               r, header='# Positions')

    np.savetxt('{}/t_final_{}_{}.txt'.format(directory,combo[0],combo[1]),
               t, header='# Time')

    fig, ax = plt.subplots()

    for j in range(chunk_length):
        if j % 10 == 0:
            plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))


    ax.set_xlim((0, 0.8))
    ax.set_ylim((0, 3.0))
    ax.legend(loc=0)
    ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
    ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
    plt.savefig('{}/vhf_final_{}_{}.pdf'.format(directory,combo[0],combo[1]))
