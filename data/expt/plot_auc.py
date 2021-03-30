import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

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

def get_auc(data, idx):
    from scipy.signal import argrelextrema
    import scipy
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

def _pairing_func(x, a, b, c, d, e, f):
    """exponential function for fitting AUC data"""
    y = a * np.exp(-(b * x)**c) + d * np.exp(-(e * x)**f)

    return y


def compute_fit(time, auc):
    time_interval = np.asarray(time)
    popt, pcov = curve_fit(_pairing_func, time_interval, auc, maxfev=5000)
    fit = _pairing_func(time_interval, *popt)

    A = popt[0]
    tau = 1 / popt[1]
    gamma = popt[2]

    #return fit, A, tau, gamma
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

expt = {
    'name': 'IXS (11/18)',
    'r': 0.1 * np.loadtxt('ixs/R_1811pure.txt')[0],
    't':  np.loadtxt('ixs/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('ixs/VHF_1811pure.txt'),
}

datas = [expt]

def first_peak_auc_rahman(datas):
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
    fig, ax = plt.subplots(figsize=(8, 8))
    columns = ('A_1', 'tau_1', 'gamma_1', 'A_2', 'tau_2', 'gamma_2')
    index = [i["name"] for i in datas]
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(1, 2):
        data = datas[i-1]
        r = data['r'] * 10 # convert from angstroms to nm
        t = data['t']
        g = data['g']

        I = np.empty_like(t)
        I[:] = np.nan

        # Get area under the curve
        #for i in range(0, t.shape[0], 2):
        for i in range(0, t.shape[0]):
            I[i] = get_auc(data, i)
        ls = '--'

        ax.semilogy(t, I, marker='.', linestyle=ls, label=data['name'])

        # Get finite values
        I_idx = np.where(~np.isnan(I))
        I = I[np.isfinite(I)]
        t = t[I_idx]
        
        print(t.shape)
        print(I.shape)
        

        #upper_limit = np.where(t < 0.28)[0][-1]
        if data["name"] != "IXS (11/18)":
            print(data["name"])
            upper_limit = np.where(t < 0.95)[0][-1]
            t = t[:upper_limit]
            I = I[:upper_limit]
            

        # Calling `compute_fit` to get the compressed exponential function fit
        try:
            fit, popt = compute_fit(t, I)
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
        ax.semilogy(t, fit, linestyle=ls, color='k', label=f"{data['name']}_fit")
        ax.set_title(data['name'], fontsize=12)

        # Plot the compressed exponential functions given from 2018 Phys. Rev.
        #A_t = 0.42*(np.exp(-(t/0.12)**1.57)) + 0.026*(np.exp(-(t/0.4)**4.1))
        #ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 310 K")
        #A_t = 0.45*(np.exp(-(t/0.12)**1.57)) + 0.018*(np.exp(-(t/0.43)**12.8))
        #ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 295 K K")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_xlim((0.00, 1.0))
        ax.set_ylim((5e-3, 1.0))
        ax.set_ylabel(r'$A(t)$')

first_peak_auc_rahman(datas)
