import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scattering
import seaborn as sns
from scattering.utils.features import find_local_maxima, find_local_minima
from scipy.signal import savgol_filter

pairs = ['O_H', 'O_O', 'H_H']

def get_color(name):
    #color_dict = {'IXS': 'black',
    #              'TIP3P': '#1f77b4',
    #              'ReaxFF': '#ff7f0e',
    #              'SPCE': '#2ca02c',
    #              'BK3': '#d62728',
    #              'DFTB_D3/3obw': '#e377c2',
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
                  'DFTB_noD3/3obw': '#937860',
                  'OptB88_filtered': '#da8bc3',
                  'AIMD': '#8c8c8c'
                 }

    return color_dict[name]

def plot_oh_peak(datas, filename, ylim=(0,3)):
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, len(datas)+1):
        ax = fig.add_subplot(2,3,i)
        data = datas[i-1]
        if data['name'] == 'OptB88_filtered':
            for j, frame in enumerate(range(len(data['t'][:1000]))):
                if j % 20 == 0:
                    ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][:1000][frame]/data['t'][:1000][-1]))
                    ax.set_title(data['name'], fontsize=12)
            norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][:1000][-1])
        else:
            for frame in range(len(data['t'][:50])):
                ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][:50][frame]/data['t'][:50][-1]))
                #ax.text(0.4, 0.1, data['name'], fontsize=12)
                ax.set_title(data['name'], fontsize=12)
            norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][:50][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        
        #ax.set_xlim((0, 0.8))
        ax.set_xlim((0.1, 0.23))
        ax.set_ylim(ylim)
        xlabel = r'$r (nm)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
        axes.append(ax)
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r'Time, ps', rotation=90)
    plt.savefig(filename)

# Plot all in one subplot
def plot_vhf_subplots(datas, filename, ylim=(0,3)):
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    cmap = matplotlib.cm.get_cmap('copper')
    for i in range(1, len(datas)+1):
        ax = fig.add_subplot(2,3,i)
        data = datas[i-1]
        for frame in range(len(data['t'])):
            if abs(data['t'][frame] % 0.1) > 0.01:
                continue
            ax.plot(data['r'], data['g'][frame], c=cmap(data['t'][frame]/data['t'][-1]))
        norm = matplotlib.colors.Normalize(vmin=data['t'][0], vmax=data['t'][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data['r'], np.ones(len(data['r'])), 'k--', alpha=0.6)
        ax.set_title(data['name'], fontsize=12)
        
        ax.set_xlim((0, 0.8))
        ax.set_ylim(ylim)
        xlabel = r'$r (nm)$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$g(r, t)$')
        axes.append(ax)
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r'Time, ps', rotation=90)
    plt.savefig(filename)

def first_peak_height(datas, filename, peak_guess=0.3, ylim=((0.06, .18)), shift=True):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=peak_guess)[1]
        if shift == True:
            ax.semilogy(data['t'], maxs-1, '--', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs, '--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.set_xlim((0.01, 0.8))
    ax.set_ylim(ylim)
    if shift == True:
        ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    else:
        ax.set_ylabel(r'$g_1(t)$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': fontsize}) 
    plt.tight_layout()
    fig.savefig(filename, dpi=500)

def first_oh_peak(datas, filename, peak_guess=0.18, ylim=((0.6, 1.8))):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        if data['name'] == 'OptB88_filtered':
            for i, frame in enumerate(data['g'][:1000]):
                maxs[i] = find_local_maxima(data['r'], frame, r_guess=peak_guess)[1]
        else:
            for i, frame in enumerate(data['g'][:50]):
                maxs[i] = find_local_maxima(data['r'], frame, r_guess=peak_guess)[1]
        ax.semilogy(data['t'], maxs, '--', lw=2, label=data['name'], color=get_color(data['name']))
    
    ax.set_xlim((0.01, 0.12))
    ax.set_ylim(ylim)
    ax.set_ylabel(r'$g_1(t)$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': fontsize}) 
    plt.tight_layout()
    fig.savefig(filename, dpi=500)

def first_peak_min(datas, filename, peak_guess=0.3):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for data in datas:    
        mins = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                mins[i] = np.nan
                continue
            mins[i] = find_local_minima(data['r'], frame, r_guess=peak_guess)[1]
        print(data['name'])
        ax.semilogy(data['t'], mins, '--', lw=2, label=data['name'], color=get_color(data['name']))
    
    #ax.set_xlim((0.01, 0.6))
    #ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'$g_{1min}(t)-1$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': fontsize})
    
    plt.tight_layout()
    fig.savefig(filename, dpi=500)

def first_second_peak(datas, filename, first_peak_guess=0.25, second_peak_guess=0.4):
    fig, ax = plt.subplots(figsize=(8, 5))
    # First peak    
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=first_peak_guess)[1]
        ax.semilogy(data['t'], maxs, '--', lw=2, label=data['name'], color=get_color(data['name']))
    # Second peak
    for data in datas:    
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=second_peak_guess)[1]
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs-1, '.-', lw=2, label=data['name'], color=get_color(data['name']))
    
    
    ax.set_xlim((0.01, 0.6))
    #ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'Peak height')
    ax.set_xlabel(r'Time, t, $ps$')
    
    #plt.legend()
    fig.savefig(filename, dpi=500)

for pair in pairs:
    aimd = {
        'r': np.loadtxt(f'../aimd/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../aimd/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../aimd/partial_data/vhf_{pair}.txt'),
        'name': 'AIMD',
    }
    
    spce = {
        'r': np.loadtxt(f'../spce/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../spce/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../spce/partial_data/vhf_{pair}.txt'),
        'name': 'SPC/E',
    }

    tip3p_ew = {
        'r': np.loadtxt(f'../../spce_vhf/tip3p_ew/1000/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../../spce_vhf/tip3p_ew/1000/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../../spce_vhf/tip3p_ew/1000/partial_data/vhf_{pair}.txt'),
        'name': 'TIP3P_EW',
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

    aimd_filtered = {
        'r': aimd['r'],
        't': aimd['t'],
        'g': savgol_filter(aimd['g'], window_length=7, polyorder=3),
        'name': 'OptB88_filtered',
    }

    #dftb = {
    #    'r': np.loadtxt('../dftb/water_form/1ns/d3_250mol_r.txt'),
    #    't': np.loadtxt('../dftb/water_form/1ns/d3_250mol_t.txt'),
    #    'g': np.loadtxt('../dftb/water_form/1ns/d3_250mol_vhf.txt'),
    #    'name': 'DFTB_D3/3obw',
    #}
    dftb = {
        'r': np.loadtxt(f'../dftb/partial_data/r_{pair}.txt'),
        't': np.loadtxt(f'../dftb/partial_data/t_{pair}.txt'),
        'g': np.loadtxt(f'../dftb/partial_data/vhf_{pair}.txt'),
        'name': 'DFTB_D3/3obw',
    }

    #datas = [tip3p, spce, bk3, reaxff, aimd]
    datas = [spce, tip3p_ew, bk3, reaxff, dftb, aimd_filtered]
    if pair == 'O_H':
        ylim = (0,2)
        peak_guess = 0.3
    elif pair == 'O_O':
        ylim = (0,3.5)
    elif pair == 'H_H':
        ylim = (0,2)
        peak_guess = 0.23

    if pair == 'O_H':
        plot_oh_peak(datas, ylim=ylim, filename=f'figures/O_H_hbond_peak.png')
        plot_oh_peak(datas, ylim=ylim, filename=f'figures/O_H_hbond_peak.pdf')
        first_oh_peak(datas,filename=f'figures/{pair}_first_peak.png')
        first_oh_peak(datas,filename=f'figures/{pair}_first_peak.pdf')

    plot_vhf_subplots(datas, ylim=ylim, filename=f'figures/{pair}_subplot.pdf')
    plot_vhf_subplots(datas, ylim=ylim, filename=f'figures/{pair}_subplot.png')
    if pair == 'H_H':
        first_peak_height(datas, peak_guess=peak_guess, ylim=(0.8, 1.8), filename=f'figures/{pair}_first_peak.pdf', shift=False)
        first_peak_height(datas, peak_guess=peak_guess, ylim=(0.8, 1.8), filename=f'figures/{pair}_first_peak.png', shift=False)
    #if pair == 'H_H':
    #    first_second_peak(datas, first_peak_guess=0.25, second_peak_guess=0.4, filename=f'figures/{pair}_first_second.pdf')
    #    first_second_peak(datas, first_peak_guess=0.25, second_peak_guess=0.4, filename=f'figures/{pair}_first_second.png')
    #if pair == 'O_H':
    #    first_peak_min(datas, peak_guess=0.2, filename=f'figures/{pair}_minima.pdf')
    #    first_peak_min(datas, peak_guess=0.2, filename=f'figures/{pair}_minima.png')
    if pair == 'O_O':
        first_peak_height(datas, peak_guess=0.3, ylim=(0.01, 2.5), filename=f'figures/{pair}_first_peak.pdf')
        first_peak_height(datas, peak_guess=0.3, ylim=(0.01, 2.5), filename=f'figures/{pair}_first_peak.png')