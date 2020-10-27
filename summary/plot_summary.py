import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#import proplot as plot

import numpy as np
import scattering
from scattering.utils.features import find_local_maxima, find_local_minima
from scipy.signal import savgol_filter

aimd = {
    'r': np.loadtxt('../aimd/water_form/r.txt'),
    't': np.loadtxt('../aimd/water_form/t.txt')*0.0005,
    'g': np.loadtxt('../aimd/water_form/vhf.txt'),
    'name': 'AIMD',
}

aimd_filtered = {
    'r': aimd['r'],
    't': aimd['t'],
    'g': savgol_filter(aimd['g'], window_length=7, polyorder=3),
    'name': 'OptB88_filtered',
}

bk3 = {
    'r': np.loadtxt('../bk3/nvt/r.txt'),
    't': np.loadtxt('../bk3/nvt/t.txt'),
    'g': np.loadtxt('../bk3/nvt/vhf.txt'),
    'name': 'BK3',
}

dftb = {
    'r': np.loadtxt('../dftb/water_form/2ns/r.txt'),
    't': np.loadtxt('../dftb/water_form/2ns/t.txt'),
    'g': np.loadtxt('../dftb/water_form/2ns/vhf.txt'),
    'name': 'DFTB_noD3/3obw',
}

dftb_d3 = {
    'r': np.loadtxt('../dftb/water_form/2ns/d3_r.txt'),
    't': np.loadtxt('../dftb/water_form/2ns/d3_t.txt'),
    'g': np.loadtxt('../dftb/water_form/2ns/d3_vhf.txt'),
    'name': 'DFTB_D3/3obw',
}

spce = {
    'r': np.loadtxt('../../spce_vhf/size_2/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/size_2/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/size_2/1000/total/vhf.txt'),
    'name': 'SPC/E',
}
#spce = {
#    'r': np.loadtxt('../../spce_vhf/spce_270/total/r.txt'),
#    't': np.loadtxt('../../spce_vhf/spce_270/total/t.txt'),
#    'g': np.loadtxt('../../spce_vhf/spce_270/total/vhf.txt'),
#    'name': 'SPC/E',
#}

reaxff = {
    'r': np.loadtxt('../reaxff/water_form/r.txt'),
    't': np.loadtxt('../reaxff/water_form/t.txt')*0.0005,
    'g': np.loadtxt('../reaxff/water_form/vhf.txt'),
    'name': 'ReaxFF',
}

tip3p = {
    'r': np.loadtxt('../../spce_vhf/tip3p/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/tip3p/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/tip3p/1000/total/vhf.txt'),
    'name': 'TIP3P',
}

tip3p_ew = {
    'r': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/vhf.txt'),
    'name': 'TIP3P_EW',
}

#IXS = {
#    'name': 'IXS',
#    'r': 0.1 * np.loadtxt('../IXS/r4Matt.txt')[0],
#    't': 0.5 * np.loadtxt('../IXS/t4Matt.txt')[:, 0],
#    'g': 1 + np.loadtxt('../IXS/VanHove4Matt.txt'),
#}
IXS = {
    'name': 'IXS',
    'r': 0.1 * np.loadtxt('../expt/R_1811pure.txt')[0],
    't': np.loadtxt('../expt/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VHF_1811pure.txt'),
}

def get_color(name):
    #color_dict = {'IXS': 'black',
    #              'TIP3P': '#1f77b4',
    #              'ReaxFF': '#ff7f0e',
    #              'SPC/E': '#2ca02c',
    #              'BK3': '#d62728',
    #              'DFTB_D3/3obw': '#e377c2',
    #              'DFTB_noD3/3obw': '#17becf',
    #              'OptB88_filtered': '#bcbd22',
    #              'AIMD': 'grey'
    #             }
    color_dict = {'IXS': 'black',
                  'TIP3P': '#4c72b0',
                  'TIP3P_EW': '#937860',
                  'ReaxFF': '#dd8452',
                  'SPC/E': '#55a868',
                  'BK3': '#c44e52',
                  'DFTB_D3/3obw': '#8172b3',
                  'OptB88_filtered': '#da8bc3',
                  'AIMD': '#8c8c8c'
                 }

    return color_dict[name]

datas = [IXS, spce, tip3p_ew, bk3, reaxff, dftb_d3, aimd_filtered]

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
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))
    max_r = list() 
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            #if data['name'] == 'DFTB_noD3/3obw':
            if data['name'] == 'SPC/E':
               max_r.append(find_local_maxima(data['r'], frame, r_guess=0.26)[0])
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': fontsize})
    plt.tight_layout()
    
    fig.savefig('figures/first_peak_height.png', dpi=500)
    fig.savefig('figures/first_peak_height.pdf', dpi=500)

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
    ax.set_xlabel('Time (ps)')
    ax.vlines(x=0.1, ymin=1e-3, ymax=2, color='k', ls='--')
    
    plt.legend(loc='upper right')
    fig.savefig('figures/auc_first_peak.png', dpi=500)
    fig.savefig('figures/auc_first_peak.pdf', dpi=500)

def second_peak(datas):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for data in datas:
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data['r'], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]
            #if data['name'] == 'SPCE':
            #    print(local_maximas[0])
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:    
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((.003, .5))
    ax.set_ylabel(r'$g_2(t)-1$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': fontsize})
    plt.tight_layout()
    fig.savefig('figures/overall_second_peak.png', dpi=500)
    fig.savefig('figures/overall_second_peak.pdf', dpi=500)

def plot_total_subplots(datas):
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    cmap = matplotlib.cm.get_cmap('copper')
    #cmap = plot.Colormap('berlin')
    #for i in range(1, 9):
    for i in range(1, 8):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i-1]
        for frame in range(len(data['t'])):
            if data['name'] == 'IXS':
                pass
            else:
                if abs(data['t'][frame] % 0.1) > 0.01:
                    continue
            ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][frame]/data['t'][-1]))

        norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        #cbar = plt.colorbar(sm)
        #cbar.set_label(r'Time, ps', rotation=90)
        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        ax.set_title(data['name'], fontsize=12)

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((round(data['r'][0], 1), round(data['r'][-1], 1)))
        ax.set_ylim((0, 3.5))

        ax.set_xlim((0, 0.8))
        xlabel = r'$r (nm)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
        axes.append(ax)
    #fig.subplots_adjust(right=0.8)
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r'Time, ps', rotation=90)
    plt.savefig('figures/total_subplot.png', dpi=500)
    plt.savefig('figures/total_subplot.pdf', dpi=500)
    
first_peak_height(datas)
#first_peak_auc(datas)
second_peak(datas)
plot_total_subplots(datas)
