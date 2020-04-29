import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scattering
from scattering.utils.features import find_local_maxima, find_local_minima

pairs = ['O_H', 'O_O', 'H_H']

# Plot all in one subplot
def plot_vhf_subplots(datas, filename, ylim=(0,3)):
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, 6):
        ax = fig.add_subplot(2,3,i)
        data = datas[i-1]
        for frame in range(len(data['t'])):
            if abs(data['t'][frame] % 0.1) > 0.01:
                continue
            ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][frame]/data['t'][-1]))
            ax.text(0.4, 0.3, data['name'], fontsize=12)
        norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label(r'Time, ps', rotation=90)
        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        
        ax.set_xlim((0, 0.8))
        ax.set_ylim(ylim)
        xlabel = r'$r$'
        if data['name'] != 'lj': xlabel += ', nm'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
    plt.tight_layout()
    plt.savefig(filename)

for pair in pairs:
    aimd = {
        'r': np.loadtxt(f'../aimd/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../aimd/partial_data/t_{pair}.txt')*.0005,
        'g': np.loadtxt(f'../aimd/partial_data/vhf_{pair}.txt'),
        'name': 'AIMD',
    }
    
    spce = {
        'r': np.loadtxt(f'../spce/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../spce/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../spce/partial_data/vhf_{pair}.txt'),
        'name': 'SPCE',
    }
    
    bk3 = {
        'r': np.loadtxt(f'../bk3/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../bk3/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../bk3/partial_data/vhf_{pair}.txt'),
        'name': 'BK3',
    }
    
    reaxff = {
        'r': np.loadtxt(f'../reaxff/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../reaxff/partial_data/t_{pair}.txt')-1000,
        'g': np.loadtxt(f'../reaxff/partial_data/vhf_{pair}.txt'),
        'name': 'ReaxFF',
    }
    
    tip3p = {
        'r': np.loadtxt(f'../tip3p/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../tip3p/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../tip3p/partial_data/vhf_{pair}.txt'),
        'name': 'TIP3P',
    }

    datas = [tip3p, spce, bk3, reaxff, aimd]
    if pair == 'O_H':
        ylim = (0,2)
    elif pair == 'O_O':
        ylim = (0,3.5)
    elif pair == 'H_H':
        ylim = (0,2)
 
    plot_vhf_subplots(datas, ylim=ylim, filename=f'figures/{pair}_subplot.pdf')
