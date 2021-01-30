import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.signal import find_peaks, argrelextrema
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter

def find_local_maxima(r, g_r, r_guess):
    """Find the local maxima nearest a guess value of r"""

    all_maxima = find_all_maxima(g_r)
    nearest_maxima, _ = find_nearest(r[all_maxima], r_guess)
    return r[all_maxima[nearest_maxima]], g_r[all_maxima[nearest_maxima]]

def find_local_minima(r, g_r, r_guess):
    """Find the local minima nearest a guess value of r"""

    all_minima = find_all_minima(g_r)
    nearest_minima, _ = find_nearest(r[all_minima], r_guess)
    return r[all_minima[nearest_minima]], g_r[all_minima[nearest_minima]]

def maxima_in_range(r, g_r, r_min, r_max):
    """Find the maxima in a range of r, g_r values"""
    idx = np.where(np.logical_and(np.greater_equal(r, r_min), np.greater_equal(r_max, r)))
    g_r_slice = g_r[idx]
    g_r_max = g_r_slice[g_r_slice.argmax()]
    idx_max, _ = find_nearest(g_r, g_r_max)
    return r[idx_max], g_r[idx_max]

def minima_in_range(r, g_r, r_min, r_max):
    """Find the minima in a range of r, g_r values"""
    idx = np.where(np.logical_and(np.greater_equal(r, r_min), np.greater_equal(r_max, r)))
    g_r_slice = g_r[idx]
    g_r_min = g_r_slice[g_r_slice.argmin()]
    idx_min, _ = find_nearest(g_r, g_r_min)
    return r[idx_min], g_r[idx_min]

def find_nearest(arr, val):
    """
    Find index in an array nearest some value.
    See https://stackoverflow.com/a/2566508/4248961
    """

    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx, arr[idx]

def find_all_minima(arr):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is greater. See https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    arr : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    checks = np.r_[True, arr[1:] < arr[:-1]] & np.r_[arr[:-1] < arr[1:], True]
    minima = np.where(checks)[0]
    return minima

def find_all_maxima(arr):
    """
    Find all local minima in a 1-D array, defined as value in which each
    neighbor is lesser. Adopted from https://stackoverflow.com/a/4625132/4248961

    Parameters
    ----------
    arr : np.ndarray
        1-D array of values

    Returns
    -------
    minima : np.ndarray
        indices of local minima
    """

    checks = np.r_[True, arr[1:] > arr[:-1]] & np.r_[arr[:-1] > arr[1:], True]
    maxima = np.where(checks)[0]
    return maxima

def get_color(name):
    color_dict = dict()
    color_list = ['TIP3P_EW', 'CHON-2017_weak', 'SPC/E', 'BK3', 'DFTB_D3/3obw', 'optB88 (filtered)',
                  'optB88 at 330K (filtered)', 'optb88', 'optB88_330K']
    colors = sns.color_palette("muted", len(color_list))
    for model, color in zip(color_list, colors):
        color_dict[model] = color 
        
    color_dict['IXS'] = 'black'

    return color_dict[name]

def get_auc(data, idx):
    from scipy.signal import argrelextrema
    import scipy
    """ Get AUC of first peak"""
    r = data['r'] * 10
    g = data['g'][idx]
    
    min1, _ = find_local_minima(r, data['g'][idx], 0.15*10)
    min2, _ = find_local_minima(r, data['g'][idx], 0.34*10) # Changed from 3.6 to 3.4
    
    # I have also tried setting the bounds to be 2.6 angstroms and 3.2 nm for all frames
    #min1 = 2.6
    #min2 = 3.2

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

def get_cn(data, idx):
    from scipy.signal import argrelextrema
    import scipy
    """ Get integral of g(r) of first peak for CN"""
    r = data['r']
    g = data['g'][idx]
    
    #min1, _ = find_local_minima(r, data['g'][idx], 0.24)
    # I have also tried setting the bounds to be 2.6 angstroms and 3.2 nm for all frames
    min1 = 0.24
    min2, _ = find_local_minima(r, data['g'][idx], 0.34) # Changed from 3.6 to 3.4
    
    # When this occurs, min2 is usually too low
    if min1 == min2:
        min2 = 0.34

    min1_idx = np.where(np.isclose(r, min1))[0][0]
    min2_idx = np.where(np.isclose(r, min2))[0][0]

    r_peak = r[min1_idx:min2_idx]
    g_peak = g[min1_idx:min2_idx]

    #auc = np.trapz(g_peak[g_peak>1] - 1, r_peak[g_peak>1])
    auc = np.trapz(g_peak*(r_peak)**2, r_peak)
    
    return auc

#def _pairing_func(x, a, b, c, d, e, f):
#    """exponential function for fitting AUC data"""
#    y = a * np.exp(-(b * x)**c) + d * np.exp(-(e * x)**f)
#
#    return y

def _pairing_func(x, a, b, d, e, f):
    """exponential function for fitting AUC data"""
    y = a * np.exp(-(b * x)**1.57) + d * np.exp(-(e * x)**f)

    return y

def compute_fit(time, auc):
    time_interval = np.asarray(time)
    bounds = ((-np.inf, 5, -np.inf, 0, -np.inf), (np.inf, np.inf, np.inf,
        3, np.inf))
    popt, pcov = curve_fit(_pairing_func, time_interval, auc, bounds=bounds, maxfev=5000)
    fit = _pairing_func(time_interval, *popt)

    return fit, popt

def compute_fit_with_guess(time, auc,guess,bounds):
    time_interval = np.asarray(time)
    popt, pcov = curve_fit(_pairing_func, time_interval, auc, p0=guess, bounds=bounds, maxfev=5000)
    fit = _pairing_func(time_interval, *popt)

    A = popt[0]
    tau = 1 / popt[1]
    gamma = popt[2]

    #return fit, A, tau, gamma
    return fit, popt

aimd = {
    'r': np.loadtxt('../aimd/water_form/r_random.txt'),
    't': np.loadtxt('../aimd/water_form/t_random.txt'),
    'g': np.loadtxt('../aimd/water_form/vhf_random.txt'),
    'name': 'optb88',
    'volume': 3.83, # nm
    'nwaters': 128,
}
aimd_330 = {
    'r': np.loadtxt('../aimd/330k/water_form/r_random.txt'),
    't': np.loadtxt('../aimd/330k/water_form/t_random.txt'),
    'g': np.loadtxt('../aimd/330k/water_form/vhf_random.txt'),
    'name': 'optB88_330K',
    'volume': 3.83, # nm
    'nwaters': 128,
}

aimd_filtered = {
    'r': aimd['r'],
    't': aimd['t'],
    'g': savgol_filter(aimd['g'], window_length=7, polyorder=3),
    'name': 'optB88 (filtered)',
    'volume': 3.83, # nm
    'nwaters': 128,
}

aimd_filtered_330 = {
    'r': aimd_330['r'],
    't': aimd_330['t'],
    'g': savgol_filter(aimd_330['g'], window_length=7, polyorder=3),
    'name': 'optB88 at 330K (filtered)',
    'volume': 3.83, # nm
    'nwaters': 128,
}

bk3 = {
    'r': np.loadtxt('../bk3/nvt/r_random.txt'),
    't': np.loadtxt('../bk3/nvt/t_random.txt'),
    'g': np.loadtxt('../bk3/nvt/vhf_random.txt'),
    'name': 'BK3',
    'volume': 30.31, # nm
    'nwaters': 1000,
}

dftb = {
    'r': np.loadtxt('../dftb/water_form/2ns/r.txt'),
    't': np.loadtxt('../dftb/water_form/2ns/t.txt'),
    'g': np.loadtxt('../dftb/water_form/2ns/vhf.txt'),
    'name': 'DFTB_noD3/3obw',
    'volume': 7.49, # nm
    'nwaters': 250,
}

dftb_d3 = {
    'r': np.loadtxt('../dftb/water_form/2ns/r_random.txt'),
    't': np.loadtxt('../dftb/water_form/2ns/t_random.txt'),
    'g': np.loadtxt('../dftb/water_form/2ns/vhf_random.txt'),
    'name': 'DFTB_D3/3obw',
    'volume': 7.49, # nm
    'nwaters': 250,
}

spce = {
    'r': np.loadtxt('../spce/total/r_random.txt'),
    't': np.loadtxt('../spce/total/t_random.txt'),
    'g': np.loadtxt('../spce/total/vhf_random.txt'),
    'name': 'SPC/E',
    'volume': 30.31, # nm
    'nwaters': 1000,
}

reaxff = {
    'r': np.loadtxt('../reaxff/water_form/r_random.txt'),
    't': np.loadtxt('../reaxff/water_form/t_random.txt'),
    'g': np.loadtxt('../reaxff/water_form/vhf_random.txt'),
    'name': 'CHON-2017_weak',
    'volume': 15.38, # nm
    'nwaters': 512,
}

tip3p_ew = {
    'r': np.loadtxt('../tip3p_ew/total/r.txt'),
    't': np.loadtxt('../tip3p_ew/total/t.txt'),
    'g': np.loadtxt('../tip3p_ew/total/vhf.txt'),
    'name': 'TIP3P_EW',
    'volume': 30.31, # nm
    'nwaters': 1000,
}


IXS = {
    'name': 'IXS',
    'r': 0.1 * np.loadtxt('../expt/R_1811pure.txt')[0],
    't': np.loadtxt('../expt/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VHF_1811pure.txt'),
}

#datas = [IXS, bk3, spce, tip3p_ew, reaxff, dftb_d3, aimd_filtered, aimd_filtered_330]

#datas = [aimd, aimd_filtered, aimd_330, aimd_filtered_330]
datas = [IXS, bk3, spce, tip3p_ew, reaxff, dftb_d3, aimd, aimd_330]
#datas = [dftb_d3]
#datas = [IXS, spce, tip3p_ew, reaxff, dftb_d3, aimd, aimd_330]

def plot_peak_locations(datas):
    """
    Function designed to replicated Fig. 2 of 2018 Phys. Rev. E paper.
    Plots the peak positions as a function of time
    
    parameters
    ----------
    datas : list
        list of dictionaries that contain VHF data
        
    returns
    -------
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # Loop through dictionaries
    for data in datas:
        r = data['r']
        
        # Get the peak height for the first peak
        # `r_low` and `r_high` are attempt to add bounds for the first peak
        r_low = np.where(r > 0.26)[0][0]
        r_high = np.where(r < 0.34)[0][-1]
        r_range = r[r_low:r_high]
        t = data['t']
        I = np.empty_like(t)
        I[:] = np.nan
        # Loop through times
        for i in range(0, t.shape[0], 5):
            g = data['g'][i][r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g, 0.28)
            print(f"R at time {t[i]} is: {r_max}")

            plt.scatter(data['t'][i], r_max, color=get_color(data["name"]), label=data["name"])
            
        # Get the peak height for the second peak
        # `r_low` and `r_high` are attempt to add bounds for the first peak
        r_low = np.where(r > 0.4)[0][0]
        r_high = np.where(r < 0.55)[0][-1]
        r_range = r[r_low:r_high]
        t = data['t']
        I = np.empty_like(t)
        I[:] = np.nan
        # Loop through times
        for i in range(0, t.shape[0], 5):
            g = data['g'][i][r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g, 0.45)

            plt.scatter(data['t'][i], r_max, color=get_color(data["name"]), label=data["name"])


    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    plt.ylabel("Peak Position (nm)")
    plt.xlabel("t (ps)")
    plt.ylim((0.25, 0.5))
    plt.xlim((0.0, 2.0))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 1.25), loc='upper center', prop={'size': 12}, ncol=4)
    plt.savefig("figures/peak_locations.pdf", dpi=500, bbox_inches="tight")
    plt.savefig("figures/peak_locations.png", dpi=500, bbox_inches="tight")

def first_peak_auc(datas):
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
    axes = list()
    columns = ('A_1', 'tau_1', 'gamma_1', 'A_2', 'tau_2', 'gamma_2')
    index = [i["name"] for i in datas]
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(1, 9):
        ax = fig.add_subplot(2, 4, i)
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
            #if data["name"] == "BK3":
            #    upper_limit = np.where(t < 1.15)[0][-1]
            #elif data["name"] == "optB88_330K":
            #    upper_limit = np.where(t < 0.9)[0][-1]
            #elif data["name"] == "optB88":
            #    upper_limit = np.where(t < 0.9)[0][-1]
            #elif data["name"] == "TIP3P_EW":
            #    upper_limit = np.where(t < 0.85)[0][-1]
            #elif data["name"] == "CHON-2017_weak":
            #    upper_limit = np.where(t < 0.75)[0][-1]
            #elif data["name"] == "DFTB_D3/3obw":
            #    upper_limit = np.where(t < 0.85)[0][-1]
            #else:
            #    #upper_limit = np.where(t < 0.95)[0][-1]
            #    upper_limit = np.where(t < 1.3)[0][-1]
            upper_limit = np.where(t < 0.90)[0][-1]
            t = t[:upper_limit]
            I = I[:upper_limit]

        # Calling `compute_fit` to get the compressed exponential function fit
        try:
            fit, popt = compute_fit(t, I)
        except:
            print(f"Fit for {data['name']} has failed")
            continue
        if (1 / popt[1]) < (1 / popt[4]):
            #df.loc[data["name"]]["A_1"] = popt[0]
            #df.loc[data["name"]]["tau_1"] = 1 / popt[1]
            #df.loc[data["name"]]["gamma_1"] = popt[2]
            #df.loc[data["name"]]["A_2"] = popt[3]
            #df.loc[data["name"]]["tau_2"] = 1 / popt[4]
            #df.loc[data["name"]]["gamma_2"] = popt[5]
            #print(data["name"])
            #print(f"tau_1 is: {1/popt[1]}")
            #print(f"A_1 is: {popt[0]}")
            #print(f"gamma_1 is: {popt[2]}")
            #print(f"tau_2 is: {1/popt[4]}")
            #print(f"A_2 is: {popt[3]}")
            #print(f"gamma_2 is: {popt[5]}")
            df.loc[data["name"]]["A_1"] = popt[0]
            df.loc[data["name"]]["tau_1"] = 1 / popt[1]
            df.loc[data["name"]]["A_2"] = popt[2]
            df.loc[data["name"]]["tau_2"] = 1 / popt[3]
            df.loc[data["name"]]["gamma_2"] = popt[4]
            print(data["name"])
            print(f"tau_1 is: {1/popt[1]}")
            print(f"A_1 is: {popt[0]}")
            print("gamma_1 is: 1.57")
            print(f"tau_2 is: {1/popt[3]}")
            print(f"A_2 is: {popt[2]}")
            print(f"gamma_2 is: {popt[4]}")
        else:
            #df.loc[data["name"]]["A_1"] = popt[3]
            #df.loc[data["name"]]["tau_1"] = 1 / popt[4]
            #df.loc[data["name"]]["gamma_1"] = popt[5]
            #df.loc[data["name"]]["A_2"] = popt[0]
            #df.loc[data["name"]]["tau_2"] = 1 / popt[1]
            #df.loc[data["name"]]["gamma_2"] = popt[2]
            #print(data["name"])
            #print(f"tau_1 is: {1/popt[4]}")
            #print(f"A_1 is: {popt[3]}")
            #print(f"gamma_1 is: {popt[5]}")
            #print(f"tau_2 is: {1/popt[1]}")
            #print(f"A_2 is: {popt[0]}")
            #print(f"gamma_2 is: {popt[2]}")
            df.loc[data["name"]]["A_1"] = popt[3]
            df.loc[data["name"]]["tau_1"] = 1 / popt[4]
            df.loc[data["name"]]["A_2"] = popt[0]
            df.loc[data["name"]]["tau_2"] = 1 / popt[1]
            df.loc[data["name"]]["gamma_2"] = popt[2]
            print(data["name"])
            print(f"tau_1 is: {1/popt[4]}")
            print(f"A_1 is: {popt[3]}")
            print(f"tau_2 is: {1/popt[1]}")
            print(f"A_2 is: {popt[0]}")
            print(f"gamma_2 is: {popt[2]}")
        ax.semilogy(t, fit, linestyle=ls, color='k', label=f"{data['name']}_fit")
        ax.set_title(data['name'], fontsize=12)

        # Plot the compressed exponential functions given from 2018 Phys. Rev.
        #A_t = 0.42*(np.exp(-(t/0.12)**1.57)) + 0.026*(np.exp(-(t/0.4)**4.1))
        #ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 310 K")
        #A_t = 0.45*(np.exp(-(t/0.12)**1.57)) + 0.018*(np.exp(-(t/0.43)**12.8))
        #ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 295 K K")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlim((0.00, 1.5))
        ax.set_ylim((5e-3, 1.0))
        #ax.set_ylim((5e-4, 1.0))
        ax.set_ylabel(r'$A(t)$')
        ax.set_xlabel('Time (ps)')

    plt.savefig("figures/first_peak_auc.pdf")
    plt.savefig("figures/first_peak_auc.png")
    df.to_csv("tables/first_peak_fits.csv")

def plot_first_peak_subplot(datas, si=False):
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
        r = data['r']
        r_low = np.where(r > 0.15)[0][0]
        r_high = np.where(r < 0.32)[0][-1]
        r_range = r[r_low:r_high]
        for i, frame in enumerate(data['g']):
            if data['t'][i] < 0.0:
                maxs[i] = np.nan
                continue
            g_range = frame[r_low:r_high]
            #maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            maxs[i] = find_local_maxima(r_range, g_range, r_guess=0.28)[1]
            if data['name'] == 'SPC/E':
               #max_r.append(find_local_maxima(data['r'], frame, r_guess=0.26)[0])
               max_r.append(find_local_maxima(r_range, g_range, r_guess=0.28)[0])
        if data['name'] == 'IXS':
            ax.semilogy(data['t'], maxs-1, '.', lw=2, label=data['name'], color=get_color(data['name']))
        else:
            ax.semilogy(data['t'], maxs-1, ls='--', lw=2, label=data['name'], color=get_color(data['name']))
    if si:
        ax.set_xlim((0.00, 1.25))
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
    else:
        ax.set_xlim((0.00, 0.6))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r'$g_1(t)-1$', fontsize=fontsize)
    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=labelsize)
    fig.legend(bbox_to_anchor=(0.45, 1.15), loc='upper center', prop={'size': fontsize}, ncol=4)

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
                color=get_color(data['name']))

        if si:
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.set_xlim((0.00, 1.25))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.set_xlim((0.00, 0.6))
        ax.set_ylim((5e-3, 1.0))
        ax.set_ylabel(r'A($t$)', fontsize=fontsize)
        ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
    if si:
        fig.savefig('figures/first_subplot_si.png', dpi=500, bbox_inches='tight')
        fig.savefig('figures/first_subplot_si.pdf', dpi=500, bbox_inches='tight')
    else:
        fig.savefig('figures/first_subplot.png', dpi=500, bbox_inches='tight')
        fig.savefig('figures/first_subplot.pdf', dpi=500, bbox_inches='tight')

def plot_first_fit():
    """ Plot compressed exponential fit for each method"""
    df = pd.read_csv("tables/first_peak_fits.csv") 
    fig, ax = plt.subplots(figsize=(6, 6))
    time = np.arange(0, 2.0, 0.05)
    for idx in range(len(df)):
        method = df.loc[idx]
        name = method[0]
        A = method["A"]
        tau = method["tau"]
        gamma = method["gamma"]

        a_t = A * np.exp(-(time/tau)**gamma)
        ax.plot(time, a_t, label=name)

        ax.set_yscale('log')

    pre = 0.448 * np.exp(-(time/0.1235)**1.57)
    ax.plot(time, pre, label="IXS, 2018 Phys. Rev.")

    ax.set_ylim(5e-3,1)
    ax.set_xlim(0,1)
    ax.set_xlabel('t/ps')
    plt.legend()
    fig.savefig('figures/tau_fit.pdf', dpi=500, bbox_inches='tight')

def first_cn(datas):
    """
    Plot the coordination number of the first peak as a function of time
    
    parameters
    ----------
    datas : list
        list of dictionaries that contain VHF data
        
    returns
    -------
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fontsize = 18
    labelsize = 18
    for data in datas:
        if data["name"] == "IXS":
            continue
        r = data['r']
        t = data['t']
        g = data['g']
        rho = 4 * np.pi * data['nwaters']/data['volume'] # molecules / nm^3

        I = np.empty_like(t)
        I[:] = np.nan
        
        # Get area under the curve
        #for i in range(0, t.shape[0], 2):
        for i in range(0, t.shape[0]):
            I[i] = get_cn(data, i) * rho
        ls = '--'

        ax.plot(t[::2], I[::2], ls=ls,
            lw=2,
            label=data['name'])

    ax.set_xlabel(r'Time, $t$, $ps$', fontsize=fontsize)
    ax.set_ylabel('Coordination Number', fontsize=fontsize)
    ax.set_xlim((0, 0.6))
    fig.savefig("figures/cn_vs_t.pdf", dpi=500, bbox_inches='tight')
        
#plot_first_fit()
first_peak_auc(datas)
#plot_peak_locations(datas)
#plot_first_peak_subplot(datas, si=True)
#plot_first_peak_subplot(datas)
#first_cn(datas)
