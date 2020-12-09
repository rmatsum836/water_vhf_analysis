import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_partial_van_hove
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
directory = 'partial_data'

combos = [['O', 'O'],
          ['O', 'H'],
          ['H', 'H']]

for combo in combos:
    r, g_r_t = compute_partial_van_hove(trj=trj_sample,
                                   chunk_length=chunk_length,
                                   selection1='element {}'.format(combo[0]),
                                   selection2='element {}'.format(combo[1]),
                                   r_range=(0, 0.8),
                                   bin_width=0.001,
                                   self_correlation=False,
                                   periodic=True)
    dt = get_dt(trj_sample)
    t = trj_sample.time[:chunk_length]

    np.savetxt('{}/vhf_{}_{}.txt'.format(directory,combo[0],combo[1]),
        g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
        dt,
        np.unique(np.round(np.diff(trj_sample.time), 6))[0],
    ))
    np.savetxt('{}/r_{}_{}.txt'.format(directory,combo[0],combo[1]),
               r, header='# Positions')
    np.savetxt('{}/t_{}_{}.txt'.format(directory,combo[0],combo[1]),
               t, header='# Time')

    fig, ax = plt.subplots()

    for j in range(chunk_length):
        plt.plot(r, g_r_t[j], label='{:.3f} ps'.format(t[j]))


    ax.set_xlim((0, 0.8))
    ax.set_ylim((0, 3.0))
    ax.legend(loc=0)
    ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
    ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
    plt.savefig('{}/van-hove-function_{}_{}.pdf'.format(directory,combo[0],combo[1]))