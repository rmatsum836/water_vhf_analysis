import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.van_hove import compute_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima


chunk_length = 200

#directory = 'water_form/2ns'
directory = 'master_test'

trj = md.load('2ns/water250_3obw-d3_100-2100ps_traject.xyz',
              top='water4vanHoveS/water4vanHove_DFTB/water4vanHove_DFTB_200-1000ps/water_250.pdb')
trj._unitcell_angles = np.zeros((trj.n_frames, 3))
trj._unitcell_lengths = np.zeros((trj.n_frames, 3))

trj._unitcell_angles[:] = [90, 90, 90]
trj._unitcell_lengths[:] = [1.9569, 1.9569, 1.9569]

trj.time = trj.time * 0.01

vhf_list = []
for start in np.random.choice(np.arange(0, trj.n_frames), 15000, replace=False):
    end = start + chunk_length
    if end > trj.n_frames:
        continue
    chunk = trj[start:end]
    print(f"Analyzing frames {start} to {end}...")
    print(chunk)
    chunk.time = np.linspace(chunk.time[0], chunk.time[-1], len(chunk.time))
    dt = get_dt(chunk)

    print('Starting vhf')
    r, t, g_r_t = compute_van_hove(trj=chunk,
                                   chunk_length=chunk_length,
                                   water=True,
                                   partial=False)
  
    vhf_list.append(g_r_t) 

t_save = t - t[0]

# Save output to text files
vhf_mean  = np.mean(vhf_list, axis=0)
#vhf_mean = np.mean(outer_list, axis=0)
np.savetxt('{}/vhf_random.txt'.format(directory),
    vhf_mean, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
    dt,
    np.unique(np.round(np.diff(trj.time), 6))[0]))
np.savetxt('{}/r_random.txt'.format(directory), r, header='# Positions')
np.savetxt('{}/t_random.txt'.format(directory), t_save, header='# Time')
#np.savetxt('{}/vhf.txt'.format(directory),
#    g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
#    dt,
#    np.unique(np.round(np.diff(trj.time), 6))[0],
#))
#np.savetxt('{}/r.txt'.format(directory), r, header='# Positions')
#np.savetxt('{}/t.txt'.format(directory), t/0.0005, header='# Time')
#np.savetxt('{}/vhf.txt'.format(directory),
#    g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
#    dt,
#    np.unique(np.round(np.diff(trj.time), 6))[0],
#))
#np.savetxt('{}/r.txt'.format(directory), r, header='# Positions')
#np.savetxt('{}/t.txt'.format(directory), t/0.0005, header='# Time')

fig, ax = plt.subplots()

for j in range(chunk_length):
    if j % 10 == 0:
        plt.plot(r, vhf_mean[j], label='{:.3f} ps'.format(t[j]))


ax.set_ylim((0, 3))
ax.legend(loc=0)
ax.set_xlabel(r'Radial coordinate, $r$, $nm$')
ax.set_ylabel(r'Van Hove Function, , $g(r, t)$, $unitiless$')
plt.savefig('{}/vhf_random.pdf'.format(directory))
