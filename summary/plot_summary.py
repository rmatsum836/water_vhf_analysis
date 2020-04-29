import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scattering
from scattering.utils.features import find_local_maxima, find_local_minima

aimd = {
    'r': np.loadtxt('../aimd/water_form/r.txt'),
    't': np.loadtxt('../aimd/no_water_form/t.txt'),
    'g': np.loadtxt('../aimd/water_form/vhf.txt'),
    'name': 'AIMD',
}

bk3 = {
    'r': np.loadtxt('../bk3/nvt/r.txt'),
    't': np.loadtxt('../bk3/nvt/t.txt'),
    'g': np.loadtxt('../bk3/nvt/vhf.txt'),
    'name': 'BK3',
}

dftb = {
    'r': np.loadtxt('../dftb/water_form/d3_r.txt'),
    't': np.loadtxt('../dftb/water_form/d3_t.txt'),
    'g': np.loadtxt('../dftb/water_form/d3_vhf.txt'),
    'name': 'DFTB_D3',
}

spce = {
    'r': np.loadtxt('../spce/nvt/r.txt'),
    't': np.loadtxt('../spce/nvt/t.txt'),
    'g': np.loadtxt('../spce/nvt/vhf.txt'),
    'name': 'SPCE',
}

reaxff = {
    'r': np.loadtxt('../reaxff/water_form/r.txt'),
    't': np.loadtxt('../reaxff/water_form/t.txt'),
    'g': np.loadtxt('../reaxff/water_form/vhf.txt'),
    'name': 'ReaxFF',
}

tip3p = {
    'r': np.loadtxt('../tip3p/nvt/r.txt'),
    't': np.loadtxt('../tip3p/nvt/t.txt'),
    'g': np.loadtxt('../tip3p/nvt/vhf.txt'),
    'name': 'TIP3P',
}

expt = {
    'name': 'expt',
    'r': 0.1 * np.loadtxt('../expt/r4Matt.txt')[0],
    't': 0.5 * np.loadtxt('../expt/t4Matt.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VanHove4Matt.txt'),
}

def get_color(name):
    color_dict = {'expt': 'black',
                  'TIP3P': '#1f77b4',
                  'ReaxFF': '#ff7f0e',
                  'SPCE': '#2ca02c',
                  'BK3': '#d62728',
                  'DFTB_D3': '#e377c2',
                  'AIMD': '#bcbd22'
                 }

    return color_dict[name]

datas = [expt, tip3p, spce, bk3, reaxff, dftb, aimd]

def get_auc(data, idx):
    r = data['r']
    g = data['g'][idx]

    min1, _ = find_local_minima(data['r'], data['g'][0], 0.25)
    min2, _ = find_local_minima(data['r'], data['g'][0], 0.35)

    auc = 0
    points = np.array(g > 1) & \
        np.array(r > min1) & \
        np.array(r < min2)

    for i, val in enumerate(points):
        if val and points[i+1]:
            r1 = r[i]
            r2 = r[i+1]
            g1 = g[i]
            g2 = g[i+1]
            auc += (r2 - r1) * ((g1 - 1) + (g2 - 1)) / 2 
        if not points[i-1] and val:
            r1 = r[i-1]
            r2 = r[i]
            g1 = g[i-1]
            g2 = g[i]
            r_ = r1 + (r2-r1)*(1-g1)/(g2-g1)
            auc += (r2 - r_) * (g2 - 1) /2
        try:
            if not points[i+1] and val:
                r1 = r[i]
                r2 = r[i+1]
                g1 = g[i]
                g2 = g[i+1]
                r_ = r1 + (r2-r1)*(1-g1)/(g2-g1)
                auc += (r_ - r1) * (g1 - 1) /2
        except IndexError:
            pass
    return auc

def first_peak_height(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
        ax.semilogy(data['t'], maxs-1, '--', lw=1, label=data['name'], color=get_color(data['name']))
    
    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((5e-2, 2))
    ax.set_ylabel(r'FIrst peak height')
    ax.set_xlabel(r'Time, t, $ps$')
    ax.vlines(x=0.1, ymin=2e-2, ymax=2, color='k', ls='--')
    
    plt.legend()
    fig.savefig('figures/first_peak_height.png', dpi=500)

def first_peak_auc(datas):
    fig, ax = plt.subplots()
    
    for data in datas:
        t = data['t']
        I = np.empty_like(t)
        I[:] = np.nan
        for i in range(data['t'].shape[0]):
            if data['name'] == 'AIMD':
                if data['t'][i] > 0.25: break
            if find_local_maxima(data['r'], data['g'][i], 0.275)[1] < 1.02:
            #if data['t'][i] > 0.3:
                break
            I[i] = get_auc(data, i)
        ls = '-'
        if data['name'] in ['TIP3P', 'ReaxFF']: ls = 'None'
        ax.semilogy(t, I, marker='.', linestyle=ls, label=data['name'].lower(), color=get_color(data['name']))
    
    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((1e-3, 1e-1))
    ax.set_ylabel(r'Area under first peak')
    ax.set_xlabel(r'Time, t, $ps$')
    ax.vlines(x=0.1, ymin=1e-3, ymax=2, color='k', ls='--')
    
    plt.legend(loc='upper right')
    fig.savefig('figures/auc_first_peak.png', dpi=500)

def second_peak(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for data in datas:
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.42)[1]
        ax.semilogy(data['t'], maxs-1, '--', lw=1, label=data['name'], color=get_color(data['name']))
    
    
    ax.set_xlim((0.01, 0.8))
    ax.set_ylim((.003, .5))
    ax.set_ylabel(r'Second peak height')
    ax.set_xlabel(r'Time, t, $ps$')
    
    plt.legend()
    fig.savefig('figures/overall_second_peak.png', dpi=500)

def plot_total_subplots(datas):
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, 8):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i-1]
        for frame in range(len(data['t'])):
            if abs(data['t'][frame] % 0.1) > 0.01:
                continue
            ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][frame]/data['t'][-1]))

        norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label(r'Time, ps', rotation=90)
        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        ax.text(0.2, 3.1, data['name'], fontsize=10)

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((round(data['r'][0], 1), round(data['r'][-1], 1)))
        ax.set_ylim((0, 3.5))

        ax.set_xlim((0, 0.8))
        xlabel = r'$r$'
        if data['name'] != 'lj': xlabel += ', nm'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
    plt.tight_layout()
    plt.savefig('figures/total_subplot.png', dpi=500)
    
first_peak_height(datas)
first_peak_auc(datas)
second_peak(datas)
plot_total_subplots(datas)
