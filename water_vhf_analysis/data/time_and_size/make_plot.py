import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt

import scattering
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scattering.utils.features import find_local_maxima, find_local_minima
from scipy.signal import savgol_filter
from scipy import optimize
from matplotlib.ticker import MultipleLocator

def get_auc(data, idx):
    from scipy.signal import argrelextrema
    import scipy
    """ Get AUC of first peak"""
    r = data['r'] * 10
    g = data['g'][idx]
    
    min1, _ = find_local_minima(r, data['g'][idx], 0.15*10)
    min2, _ = find_local_minima(r, data['g'][idx], 0.34*10) # Changed from 3.6 to 3.4
    
    # When this occurs, min2 is usually too low
    if min1 == min2:
        min2 = 0.34 * 10

    #min1_idx = np.where(r == min1)[0][0]
    min1_idx = np.where(np.isclose(r, min1, rtol=0.02))[0][0]
    min2_idx = np.where(np.isclose(r, min2, rtol=0.02))[0][0]

    r_peak = r[min1_idx:min2_idx]
    g_peak = g[min1_idx:min2_idx]

    auc = np.trapz(g_peak[g_peak>1] - 1, r_peak[g_peak>1])
    
    return auc

def _pairing_func(x, a, b, c, d, e, f):
    """exponential function for fitting AUC data"""
    y = a * np.exp(-(b * x)**c) + d * np.exp(-(e * x)**f)

    return y

def compute_fit(time, auc):
    time_interval = np.asarray(time)
    bounds = ((-np.inf, 5, 0, -np.inf, 0, -np.inf), (np.inf, np.inf, 10, np.inf,
        10, np.inf))
    popt, pcov = curve_fit(_pairing_func, time_interval, auc, bounds=bounds, maxfev=5000)
    fit = _pairing_func(time_interval, *popt)

    return fit, popt

expt = {
    'name': 'IXS',
    'r': 0.1 * np.loadtxt('../expt/R_1811pure.txt')[0],
    't':  np.loadtxt('../expt/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VHF_1811pure.txt'),
}

sys_2ns = {
    'name': '2 ns',
    'r': np.loadtxt('time/2ns/overlap_nvt/r_final.txt'),
    't': np.loadtxt('time/2ns/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('time/2ns/overlap_nvt/vhf_final.txt'),
}

sys_1ns = {
    'name': '1 ns',
    'r': np.loadtxt('time/1ns/overlap_nvt/r_final.txt'),
    't': np.loadtxt('time/1ns/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('time/1ns/overlap_nvt/vhf_final.txt'),
}

sys_100ps = {
    'name': '100 ps', 
    'r': np.loadtxt('time/100ps/overlap_nvt/r_final.txt'),
    't': np.loadtxt('time/100ps/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('time/100ps/overlap_nvt/vhf_final.txt'),
}


sys_50ps = {
    'name': '50 ps', 
    'r': np.loadtxt('time/50ps/overlap_nvt/r_final.txt'),
    't': np.loadtxt('time/50ps/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('time/50ps/overlap_nvt/vhf_final.txt'),
}

sys_500ps = {
    'name': '500 ps',
    'r': np.loadtxt('time/500ps/overlap_nvt/r_final.txt'),
    't': np.loadtxt('time/500ps/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('time/500ps/overlap_nvt/vhf_final.txt'),
}

sys_128 = {
    'name': "128",
    'r': np.loadtxt('size/128/overlap_nvt/r_final.txt'),
    't': np.loadtxt('size/128/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('size/128/overlap_nvt/vhf_final.txt'),
}

sys_250 = {
    'name': "250",
    'r': np.loadtxt('size/250/overlap_nvt/r_final.txt'),
    't': np.loadtxt('size/250/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('size/250/overlap_nvt/vhf_final.txt'),
}

sys_512 = {
    'name': "512",
    'r': np.loadtxt('size/512/overlap_nvt/r_final.txt'),
    't': np.loadtxt('size/512/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('size/512/overlap_nvt/vhf_final.txt'),
}

sys_1000 = {
    'name': "1000",
    'r': np.loadtxt('size/1000/overlap_nvt/r_final.txt'),
    't': np.loadtxt('size/1000/overlap_nvt/t_final.txt'),
    'g': np.loadtxt('size/1000/overlap_nvt/vhf_final.txt'),
}
datas = [expt, sys_50ps, sys_100ps, sys_500ps, sys_1ns, sys_2ns]
size_datas = [expt, sys_128, sys_250, sys_512, sys_1000]
size_datas_no_ixs = [sys_128, sys_250, sys_512, sys_1000]

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def first_peak_height(datas):
    fig, ax = plt.subplots(figsize=(9, 5))
    fontsize = 14
    labelsize = 14
    
    for data in datas:
        maxs = np.zeros(len(data['t']))
        gauss_maxs = np.zeros(len(data['t']))

        r_low = np.where(data['r'] > 0.20)[0][0]
        r_high = np.where(data['r'] < 0.34)[0][-1]
        r_range = data['r'][r_low:r_high]
        for i, frame in enumerate(data['g']):
            g_range = data['g'][i][r_low:r_high]
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            if data["name"] == "100ps":
                try:
                    popt, _ = optimize.curve_fit(gaussian, r_range, g_range)
                    gauss_fit = gaussian(r_range, *popt)
                    gauss_maxs[i] = find_local_maxima(r_range, gauss_fit, r_guess=0.26)[1]
                except:
                    continue
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=1, label=data['name'], color='k')
        elif data['name'] == '100ps':
            ax.semilogy(data['t'], gauss_maxs-1, '--', lw=2, label=data['name'])
        else:
            ax.semilogy(data['t'], maxs-1, '--', lw=2, label=data['name'])

    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((5e-2, 2))
    ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel('Time (ps)', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.legend(prop={'size': fontsize})
    plt.tight_layout()
    plt.savefig('plots/first_peak.pdf', dpi=500) 
    plt.savefig('plots/first_peak.png', dpi=500) 

def combined_with_size(datas, size_datas):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fontsize = 14
    labelsize = 14
   
    # Plot size data 
    ax = axes[0]
    ax.text(-0.10, 0.90, 'a)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in size_datas:
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color='k')
        else:
            ax.semilogy(data['t'], maxs-1, '--', lw=2, label=data['name'])

    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((5e-2, 2))
    ax.set_ylabel(r'$G_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.legend(prop={'size': fontsize})

    # Plot time data
    ax = axes[1]
    ax.text(-0.10, 0.90, 'b)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in datas:
        maxs = np.zeros(len(data['t']))
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=1, label=data['name'], color='k')
        else:
            ax.semilogy(data['t'], maxs-1, '--', lw=2, label=data['name'])

    ax.set_xlim((0.01, 0.6))
    ax.set_ylim((5e-2, 2))
    ax.set_ylabel(r'$G_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.legend(prop={'size': fontsize})
    plt.tight_layout()
    plt.savefig('plots/time_and_size_peak.pdf', dpi=500) 
    plt.savefig('plots/time_and_size_peak.png', dpi=500) 

def first_subplot(datas):
    fontsize = 14
    labelsize = 14
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    for i in range(1, 6):
        ax = fig.add_subplot(2, 3, i)
        all_data = datas[i-1]
        raw = all_data[0]
        filtered = all_data[1]
        for data in (raw, filtered): 
            maxs = np.zeros(len(data['t']))
            gauss_maxs = np.zeros(len(data['t']))

            r_low = np.where(data['r'] > 0.20)[0][0]
            r_high = np.where(data['r'] < 0.34)[0][-1]
            r_range = data['r'][r_low:r_high]
            for i, frame in enumerate(data['g']):
                g_range = data['g'][i][r_low:r_high]
                if data['t'][i] < 0.0:
                    maxs[i] = np.nan
                    continue
                maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            if "filtered" in data["name"]:
                ax.semilogy(data['t'][::2], (maxs-1)[::2], color='k', marker='.', ls=None, label=data['name'])
            else:
                ax.semilogy(data['t'], (maxs-1), '--', lw=2, label=data['name'])

        ax.set_title(data['name'], fontsize=12)
        ax.set_xlim((0.01, 0.6))
        ax.set_ylim((5e-2, 2))
        ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
        ax.set_xlabel('Time (ps)', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)

    plt.tight_layout()
    plt.savefig('plots/first_subplot_size.pdf', dpi=500) 
    plt.savefig('plots/first_subplot_size.png', dpi=500) 

def combined_auc(datas, size_datas):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fontsize = 14
    labelsize = 14
   
    # Plot size data 
    ax = axes[0]
    ax.text(-0.10, 0.90, 'a)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in size_datas:
        r = data['r'] * 10 # convert from angstroms to nm
        t = data['t']
        g = data['g']

        I = np.empty_like(t)
        I[:] = np.nan
        
        # Get area under the curve
        for i in range(0, t.shape[0], 2):
            I[i] = get_auc(data, i)

        ls = '--'
        if data['name'] == 'IXS':
            ax.semilogy(t, I, marker='.',
                label=data['name'],
                color='k')
        else: 
            # Get rid of NANs with [::2]
            ax.semilogy(t[::2], I[::2], ls=ls,
                lw=2,
                label=data['name'],
                )

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlim((0.00, 0.6))
    ax.set_ylim((5e-3, 1.0))
    ax.set_ylabel(r'A($t$)', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.legend(prop={'size': fontsize})

    # Plot time data
    ax = axes[1]
    ax.text(-0.10, 0.90, 'b)', transform=ax.transAxes,
            size=20, weight='bold')
    for data in datas:
        r = data['r'] * 10 # convert from angstroms to nm
        t = data['t']
        g = data['g']

        I = np.empty_like(t)
        I[:] = np.nan
        
        # Get area under the curve
        for i in range(0, t.shape[0], 2):
            I[i] = get_auc(data, i)

        ls = '--'
        if data['name'] == 'IXS':
            ax.semilogy(t, I, marker='.',
                label=data['name'],
                color='k')
        else: 
            # Get rid of NANs with [::2]
            ax.semilogy(t[::2], I[::2], ls=ls,
                lw=2,
                label=data['name'],
                )

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlim((0.00, 0.6))
    ax.set_ylim((5e-3, 1.0))
    ax.set_ylabel(r'A($t$)', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)


    ax.legend(prop={'size': fontsize})
    plt.tight_layout()
    plt.savefig('plots/auc_time_and_size.pdf', dpi=500) 
    plt.savefig('plots/auc_time_and_size.png', dpi=500) 

def first_peak_auc(time_datas, size_datas):
    """
    Plot the AUC of first peak as a function of time
    Attempting to replicate Fig. 4b of Phys. Rev. E
    
    parameters
    ----------
    datas : list
        list of dictionaries that contain VHF data
        
    returns
    -------
    """
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    datas = time_datas + size_datas
    axes = list()
    columns = ('A_1', 'tau_1', 'gamma_1', 'A_2', 'tau_2', 'gamma_2')
    index = [i["name"] for i in datas]
    print(index)
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(1, 11):
        ax = fig.add_subplot(2, 5, i)
        data = datas[i-1]
        r = data['r'] * 10 # convert from angstroms to nm
        t = data['t']
        g = data['g']

        I = np.empty_like(t)
        I[:] = np.nan
        
        # Get area under the curve
        for i in range(0, t.shape[0]):
            I[i] = get_auc(data, i)
        ls = '--'

        ax.semilogy(t[::2], I[::2], ls=ls,
            lw=2,
            label=data['name'])
        
        # Get finite values
        I_idx = np.where(~np.isnan(I))
        I = I[np.isfinite(I)]
        t = t[I_idx]

        #upper_limit = np.where(t < 0.28)[0][-1]
        if data["name"] not in ("IXS"):
            if data["name"] == "optB88":
                upper_limit = np.where(t < 1.15)[0][-1]
            elif data["name"] == "CHON-2017_weak":
                upper_limit = np.where(t < 0.75)[0][-1]
            elif data["name"] == "DFTB_D3/3obw":
                upper_limit = np.where(t < 0.9)[0][-1]
            else:
                #upper_limit = np.where(t < 0.90)[0][-1]
                upper_limit = np.where(t < 1.00)[0][-1]
            t = t[:upper_limit]
            I = I[:upper_limit]

        # Calling `compute_fit` to get the compressed exponential function fit
        try:
            print(len(t[::2]))
            print(len(I[::2]))
            fit, popt = compute_fit(t[::2], I[::2])
        except:
            print(f"Fit for {data['name']} has failed")
            continue
        if (1 / popt[1]) < (1 / popt[4]):
            df.loc[data["name"]]["A_1"] = popt[0]
            df.loc[data["name"]]["tau_1"] = 1 / popt[1]
            df.loc[data["name"]]["gamma_1"] = popt[2]
            df.loc[data["name"]]["A_2"] = popt[3]
            df.loc[data["name"]]["tau_2"] = 1 / popt[4]
            df.loc[data["name"]]["gamma_2"] = popt[5]
            print(data["name"])
            print(f"tau_1 is: {1/popt[1]}")
            print(f"A_1 is: {popt[0]}")
            print(f"gamma_1 is: {popt[2]}")
            print(f"tau_2 is: {1/popt[4]}")
            print(f"A_2 is: {popt[3]}")
            print(f"gamma_2 is: {popt[5]}")
        else:
            df.loc[data["name"]]["A_1"] = popt[3]
            df.loc[data["name"]]["tau_1"] = 1 / popt[4]
            df.loc[data["name"]]["gamma_1"] = popt[5]
            df.loc[data["name"]]["A_2"] = popt[0]
            df.loc[data["name"]]["tau_2"] = 1 / popt[1]
            df.loc[data["name"]]["gamma_2"] = popt[2]
            print(data["name"])
            print(f"tau_1 is: {1/popt[4]}")
            print(f"A_1 is: {popt[3]}")
            print(f"gamma_1 is: {popt[5]}")
            print(f"tau_2 is: {1/popt[1]}")
            print(f"A_2 is: {popt[0]}")
            print(f"gamma_2 is: {popt[2]}")
        ax.semilogy(t[::2], fit, linestyle=ls, color='k', label=f"{data['name']}_fit")
        ax.set_title(data['name'], fontsize=12)

        # Plot the compressed exponential functions given from 2018 Phys. Rev.
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlim((0.00, 1.0))
        ax.set_ylim((5e-3, 1.0))
        ax.set_ylabel(r'$A(t)$')
        ax.set_xlabel('Time (ps)')

    df.to_csv("tables/time_and_size_peak_fits.csv")
    plt.savefig("plots/auc_subplot_time_size.png", dpi=500)

#first_peak_height(datas)
#first_peak_auc(datas)
#combined_with_size(datas, size_datas)
combined_auc(datas, size_datas)
first_peak_auc(datas, size_datas_no_ixs)
#first_subplot(datas)
