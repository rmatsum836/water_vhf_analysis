import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import numpy as np
import scattering
from scattering.utils.features import find_local_maxima, find_local_minima
from scipy.signal import savgol_filter
from matplotlib.ticker import MultipleLocator
from scipy import optimize

aimd = {
    'r': np.loadtxt('../aimd/water_form/r.txt'),
    't': np.loadtxt('../aimd/water_form/t.txt')[::20]*0.0005,
    'g': np.loadtxt('../aimd/water_form/vhf.txt')[::20],
    'name': 'optB88',
}

aimd_330 = {
    'r': np.loadtxt('../aimd/330k/water_form/r.txt'),
    't': np.loadtxt('../aimd/330k/water_form/t.txt')[::10]*0.0005,
    'g': np.loadtxt('../aimd/330k/water_form/vhf.txt')[::10],
    'name': 'optB88_330K',
}

#aimd_filtered = {
#    'r': aimd['r'],
#    't': aimd['t'],
#    'g': savgol_filter(aimd['g'], window_length=7, polyorder=3),
#    'name': 'optB88 (filtered)',
#}
aimd_filtered = {
    'r': aimd['r'],
    't': aimd['t'],
    'g': savgol_filter(aimd['g'], window_length=11, polyorder=4),
    'name': 'optB88 (filtered)',
}

aimd_filtered_330 = {
    'r': aimd_330['r'],
    't': aimd_330['t'],
    'g': savgol_filter(aimd_330['g'], window_length=7, polyorder=3),
    'name': 'optB88 at 330K (filtered)',
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

dftb_filtered = {
    'r': dftb_d3['r'],
    't': dftb_d3['t'],
    'g': savgol_filter(dftb_d3['g'], window_length=7, polyorder=2),
    'name': 'DFTB_D3 (filtered)',
}

spce = {
    'r': np.loadtxt('../../spce_vhf/size_2/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/size_2/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/size_2/1000/total/vhf.txt'),
    'name': 'SPC/E',
}

reaxff = {
    'r': np.loadtxt('../reaxff/water_form/r.txt'),
    't': np.loadtxt('../reaxff/water_form/t.txt')*0.0005,
    'g': np.loadtxt('../reaxff/water_form/vhf.txt'),
    'name': 'CHON-2017_weak',
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

IXS = {
    'name': 'IXS',
    'r': 0.1 * np.loadtxt('../expt/R_1811pure.txt')[0],
    't': np.loadtxt('../expt/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VHF_1811pure.txt'),
}

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def get_color(name):
    color_dict = dict()
    color_list = ['TIP3P_EW', 'CHON-2017_weak', 'SPC/E', 'BK3', 'DFTB_D3/3obw', 'optB88 (filtered)',
                  'optB88 at 330K (filtered)', 'optB88', 'DFTB_D3 (filtered)', 'optB88_330K']
    colors = sns.color_palette("muted", len(color_list))
    for model, color in zip(color_list, colors):
        color_dict[model] = color 
        
    color_dict['IXS'] = 'black'

    return color_dict[name]

datas = [IXS, spce, tip3p_ew, bk3, reaxff, dftb_d3, aimd_filtered, aimd_filtered_330]
#datas = [IXS, spce, tip3p_ew, bk3, reaxff, aimd_filtered, dftb_filtered]
#datas = [aimd, aimd_filtered, aimd_330, aimd_filtered_330]
#datas = [IXS, spce, tip3p_ew, bk3, reaxff, dftb_d3]

def make_heatmap(data, ax, v=0.1, fontsize=14):
    heatmap = ax.imshow(
        data['g'] - 1,
        vmin=-v, vmax=v,
        cmap='viridis',
        origin='lower',
        aspect='auto',
        extent=(data['r'][0], data['r'][-1], data['t'][0], data['t'][-1])
    )
    ax.grid(False)
    ax.set_xlim((round(data['r'][0], 1), 0.8))
    ax.set_ylim((0, 1))#ax.set_ylim((round(data['t'][0], 1), round(data['t'][-1], 2)))

    xlabel = r'r, $nm$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    ax.set_title(data['name'], fontsize=fontsize, y=1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))

    return heatmap

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
        # Get r_range for getting gaussian fits
        r_low = np.where(data["r"] > 0.20)[0][0]
        r_high = np.where(data["r"] < 0.34)[0][-1]
        r_range = data["r"][r_low:r_high]

        maxs = np.zeros(len(data['t']))
        gauss_maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            #if data["name"] in ("AIMD", "DFTB_D3/3obw"):
            if data["name"] in ("AIMD"):
                g_range = data["g"][i][r_low:r_high]
                try:
                    popt, _ = optimize.curve_fit(gaussian, r_range, g_range)
                    gauss_fit = gaussian(r_range, *popt)
                    gauss_maxs[i] = find_local_maxima(r_range, gauss_fit, r_guess=0.26)[1]
                except:
                    continue
            #if data['name'] == 'DFTB_noD3/3obw':
            if data['name'] == 'SPC/E':
               max_r.append(find_local_maxima(data['r'], frame, r_guess=0.26)[0])
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        #elif data['name'] in ("AIMD", "DFTB_D3/3obw"):
        elif data['name'] in ("AIMD"):
            ax.semilogy(data['t'], gauss_maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', prop={'size': fontsize}, ncol=4)
    plt.tight_layout()
    
    fig.savefig('figures/first_peak_height.png', dpi=500, bbox_inches="tight")
    fig.savefig('figures/first_peak_height.pdf', dpi=500, bbox_inches="tight")

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
        if data['name'] in ['TIP3P', 'CHON-2017_weak']: ls = 'None'
        ax.semilogy(t, I, marker='.', linestyle=ls, label=data['name'].lower(), color=get_color(data['name']))
    
    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((1e-3, 1e-1))
    ax.set_ylabel(r'Area under first peak')
    ax.set_xlabel('Time (ps)')
    ax.vlines(x=0.1, ymin=1e-3, ymax=2, color='k', ls='--')
    
    plt.legend(loc='upper right')
    fig.savefig('figures/auc_first_peak.png', dpi=500)
    fig.savefig('figures/auc_first_peak.pdf', dpi=500)

def second_peak(datas, normalize=False):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for data in datas:
        print(data['name'])
        # Find nearest value to 0.1
        if normalize:
            norm_idx = find_nearest(data['t'], 0.25)
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data['r'], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        if normalize:
            if data['name'] == 'IXS':
                ax.plot(data['t'], (maxs-1)/(maxs[0]-1), '.', lw=2, label=data['name'], color=get_color(data['name']))
            else:    
                ax.plot(data['t'], (maxs-1)/(maxs[0]-1), ls='--', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            if data['name'] == 'IXS':
                ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
            else:    
                ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.tick_params(axis='both', labelsize=labelsize)
    
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', prop={'size': fontsize}, ncol=4)
    plt.tight_layout()
    if normalize:
        ax.set_xlim((0.005, 0.8))
        ax.set_ylim((0.0, 1.5))
        ax.set_ylabel(r'$g_2(t) / g_2(0)$, normalized', fontsize=fontsize)
        ax.set_xlabel(r'Time, $t$ / $t(0)$, $ps$', fontsize=fontsize)
        fig.savefig('figures/overall_second_peak_normalize.png', dpi=500, bbox_inches='tight')
        fig.savefig('figures/overall_second_peak_normalize.pdf', dpi=500, bbox_inches='tight')
    else:
        ax.set_xlim((0.005, 0.8))
        ax.set_ylim((.01, .5))
        ax.set_ylabel(r'$g_2(t)-1$', fontsize=fontsize)
        ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
        fig.savefig('figures/overall_second_peak.png', dpi=500, bbox_inches='tight')
        fig.savefig('figures/overall_second_peak.pdf', dpi=500, bbox_inches='tight')

def plot_total_subplots(datas):
    fontsize = 16

    fig = plt.figure(figsize=(20, 14))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    axes = list()
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, 9):
        ax = fig.add_subplot(4, 4, i)
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
        ax.set_title(data['name'], fontsize=fontsize, y=1.05)

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((round(data['r'][0], 1), round(data['r'][-1], 1)))
        ax.set_ylim((0, 3.5))

        ax.set_xlim((0, 0.8))
        xlabel = r'r, $nm$'
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(r'$g(r, t)$', fontsize=fontsize)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        axes.append(ax)
    cbar = fig.colorbar(sm, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r'Time, $t$, $ps$', rotation=90, fontsize=fontsize)

    axes = list()
    for i in range(9, 17):
        ax = fig.add_subplot(4, 4, i)
        data = datas[(i-8)-1]
        heatmap = make_heatmap(data, ax, fontsize=fontsize)
        axes.append(ax)

    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r'$g(r, t) - 1$', rotation=90, fontsize=fontsize)
    plt.savefig('figures/total_subplot.png', bbox_inches='tight', dpi=500)
    plt.savefig('figures/total_subplot.pdf', bbox_inches='tight', dpi=500)

def plot_self_subplots(datas):
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, 8):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i-1]
        cutoff_max = np.where(np.isclose(data['t'], 1.0 , 0.02))[0][0]
        cutoff_min = np.where(np.isclose(data['t'], 0.1 , 0.05))[0][0]
        for idx, (frame, g_r) in enumerate(zip(data['t'][cutoff_min:cutoff_max], data['g'][cutoff_min:cutoff_max])):
            if data['name'] == 'IXS':
                pass
            #else:
            #    if data['t'][frame] < 0.2:
            #        continue
            if data['name'] == 'optB88_filtered':
                if idx % 5 != 0:
                    continue
            ax.plot(data['r'], g_r, c=cmap(frame/data['t'][cutoff_max]))

        norm = matplotlib.colors.Normalize(vmin=data['t'][cutoff_min], vmax=data['t'][cutoff_max])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        ax.set_title(data['name'], fontsize=12)

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((0, 0.2))
        ax.set_ylim((0, 200.0))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        xlabel = r'r, $nm$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
        axes.append(ax)
    #fig.subplots_adjust(right=0.8)
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r'Time, $t$, $ps$', rotation=90, fontsize=14)
    plt.savefig('figures/self_subplot.png', dpi=500)
    plt.savefig('figures/self_subplot.pdf', dpi=500)

def plot_heatmap(datas):
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    for i in range(1, 9):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i-1]
        heatmap = make_heatmap(data, ax)
        axes.append(ax)

    plt.tight_layout()
    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.set_label(r'$g(r, t) - 1$', rotation=90, fontsize=14)
    plt.savefig('figures/heatmap.png', dpi=500)
    plt.savefig('figures/heatmap.pdf', dpi=500)

def plot_decay_subplot(datas):
    fontsize = 18
    labelsize = 18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("muted", len(datas))

    # Plot first peak decay
    ax = axes[0]
    ax.text(-0.10, 0.90, 'a)', transform=ax.transAxes,
            size=20, weight='bold')
    ax.set_prop_cycle('color', colors)
    max_r = list() 
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            if data['name'] == 'SPC/E':
               max_r.append(find_local_maxima(data['r'], frame, r_guess=0.26)[0])
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        # Grabbing every fifth data point of AIMD data
        elif data['name'] in ('optB88 (filtered)', 'optB88 at 330K (filtered)', 'AIMD'):
            ax.semilogy(data['t'][::5], (maxs-1)[::5], ls='--', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    ax.set_xlim((0.00, 0.6))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    fig.legend(bbox_to_anchor=(0.45, 1.15), loc='upper center', prop={'size': fontsize}, ncol=4)

    # Plot second peak decay
    ax = axes[1]
    ax.text(-0.10, 0.90, 'b)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in datas:
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data['r'], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        elif data['name'] in ('optB88 (filtered)', 'optB88 at 330K (filtered)', 'AIMD'):
            ax.semilogy(data['t'][::5], (maxs-1)[::5], ls='--', lw=2, label=data['name'], color=get_color(data['name']))
        else:    
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.set_xlim((0.00, 0.8))
    #ax.set_ylim((.003, .5))
    ax.set_ylim((.01, .5))
    ax.set_ylabel(r'$g_2(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', labelsize=labelsize)
    
    fig.savefig('figures/peak_decay.png', dpi=500, bbox_inches='tight')
    fig.savefig('figures/peak_decay.pdf', dpi=500, bbox_inches='tight')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_second_subplot(datas):
    fontsize = 18
    labelsize = 18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("muted", len(datas))
    
    ax = axes[0]
    ax.text(-0.10, 1.0, 'a)', transform=ax.transAxes,
            size=20, weight='bold')
    ax.set_prop_cycle('color', colors)
    for data in datas:
        print(data['name'])
        # Find nearest value to 0.1
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data['r'], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:    
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((.01, .5))
    ax.set_ylabel(r'$g_2(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    fig.legend(bbox_to_anchor=(0.45, 1.15), loc='upper center', prop={'size': fontsize}, ncol=4)
    
    ax = axes[1]
    ax.text(-0.10, 1.0, 'b)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in datas:
        print(data['name'])
        # Find nearest value to 0.1
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data['r'], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        min_max = ((maxs-1)-np.min(maxs-1)) / (np.max(maxs-1)-np.min(maxs-1))
        if data['name'] == 'IXS':
            #ax.plot(data['t'], (maxs-1)/(maxs[0]-1), '.', lw=2, label=data['name'], color=get_color(data['name']))
            ax.plot(data['t'], min_max, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:    
            #ax.plot(data['t'], (maxs-1)/(maxs[0]-1), ls='--', lw=2, label=data['name'], color=get_color(data['name']))
            ax.plot(data['t'], min_max, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((0.0, 1.10))
    #ax.set_ylabel(r'$g_2(t) / g_2(0)$, normalized', fontsize=fontsize)
    ax.set_ylabel(r'$g_2(t)-1$, normalized', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)

    fig.savefig('figures/second_subplot.png', dpi=500, bbox_inches='tight')
    fig.savefig('figures/second_subplot.pdf', dpi=500, bbox_inches='tight')
    
#first_peak_height(datas)
#first_peak_auc(datas)
#second_peak(datas, normalize=True)
#plot_total_subplots(datas)
#plot_self_subplots(datas)
#plot_heatmap(datas)
#plot_decay_subplot(datas)
plot_second_subplot(datas)
