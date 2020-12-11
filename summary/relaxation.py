import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scattering
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, argrelextrema
from scattering.utils.features import find_local_maxima, find_local_minima


spce = {
    'r': np.loadtxt('../../spce_vhf/size_2/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/size_2/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/size_2/1000/total/vhf.txt'),
    'name': 'SPC/E',
}

bk3 = {
    'r': np.loadtxt('../bk3/nvt/r.txt'),
    't': np.loadtxt('../bk3/nvt/t.txt'),
    'g': np.loadtxt('../bk3/nvt/vhf.txt'),
    'name': 'BK3',
}

reaxff = {
    'r': np.loadtxt('../reaxff/water_form/r.txt'),
    't': np.loadtxt('../reaxff/water_form/t.txt')*0.0005,
    'g': np.loadtxt('../reaxff/water_form/vhf.txt'),
    'name': 'CHON-2017_weak',
}

tip3p_ew = {
    'r': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/r.txt'),
    't': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/t.txt'),
    'g': np.loadtxt('../../spce_vhf/tip3p_ew/1000/total/vhf.txt'),
    'name': 'TIP3P_EW',
}

dftb_d3 = {
    'r': np.loadtxt('../dftb/water_form/2ns/d3_r.txt'),
    't': np.loadtxt('../dftb/water_form/2ns/d3_t.txt'),
    'g': np.loadtxt('../dftb/water_form/2ns/d3_vhf.txt'),
    'name': 'DFTB_D3/3obw',
}

IXS = {
    'name': 'IXS',
    'r': 0.1 * np.loadtxt('../expt/R_1811pure.txt')[0],
    't': np.loadtxt('../expt/t_1811pure.txt')[:, 0],
    'g': 1 + np.loadtxt('../expt/VHF_1811pure.txt'),
}


datas = [spce, bk3, reaxff, tip3p_ew, dftb_d3, IXS]

# TODO: Instead of using the trapezoidal rule, integrate over G(r, t)

def plot_first_peak(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    for data in datas:
        r = data['r']
        r_low = np.where(r > 0.2)[0][0]
        r_high = np.where(r < 0.35)[0][-1]
        r_range = r[r_low:r_high]
        t = data['t']

        I = np.empty_like(t)
        I[:] = np.nan
        for i in range(0, t.shape[0], 5):
            if t[i] > 1.5:
                continue
            g = data['g'][i][r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g, 0.3)

            plt.scatter(data['t'][i], r_max, color='k')

    plt.ylabel("Peak Position (nm)")
    plt.xlabel("t (ps)")
    plt.ylim((0.25, 0.5))

    plt.savefig("maxima.pdf")

def first_peak_auc(datas, calc_fit=True):
    """ Plot AUC of first peak

    Parameters
    ----------
    datas : list
        List of dictionaries containing VHF data
    calc_fit : bool, default=True
        If true, calculate fit of AUC curves

    """
    fig, ax = plt.subplots(figsize=(4, 5))
    for data in datas:
        print(data['name']) 
        r = data['r'] * 10
        t = data['t']
        g = data['g']

        I = np.empty_like(t)
        I[:] = np.nan
        #for i in range(0, t.shape[0], 5):
        for i in range(0, t.shape[0]):
            I[i] = get_auc(data, i)
        ls = '-'
        if data['name'] in ['TIP3P', 'CHON-2017_weak']:
            ls = 'None'

        ax.semilogy(t, I, marker='.', linestyle=ls, label=data['name'].lower())

        if calc_fit == True:
            # Get finite values
            I = I[np.isfinite(I)]
            t = t[:len(I)]
            # Get fits for both steps of decay
            for i in range(1, 2):
                if i == 1:
                    I_range = I[:20]
                    t_range = t[:20]
                elif i == 2:
                    I_range = I[30:50]
                    t_range = t[30:50]

                fit, A, tau, gamma = compute_fit(t_range, I_range)
                print(f"tau_{i} is: {tau}")
                print(f"A_{i} is: {A}")
                print(f"gamma_{i} is: {gamma}")
                ax.semilogy(t_range, fit, linestyle=ls, label=f"{data['name'].lower()} (fit)")
                #ax.plot(t_range, fit, linestyle=ls, label=data['name'].lower())
    
    ax.set_xlim((0.00, 1.0))
    ax.set_ylim((5e-3, 1))
    ax.set_ylabel(r'Area under first peak')
    ax.set_xlabel('Time (ps)')
    ax.vlines(x=0.1, ymin=1e-3, ymax=2, color='k', ls='--')
    #ax.set_title(f"tau: {tau}, A: {A}, gamma: {gamma}")
    plt.legend()

    plt.savefig("tau_fitting.pdf", bbox_inches='tight')

def get_auc(data, idx):
    """ Get AUC of first peak"""
    r = data['r'] * 10
    g = data['g'][idx]

    min1, _ = find_local_minima(r, data['g'][0], 0.25*10)
    min2, _ = find_local_minima(r, data['g'][0], 0.35*10)

    min1_idx = np.where(r == min1)[0][0]
    min2_idx = np.where(r == min2)[0][0]

    r_peak = r[min1_idx:min2_idx]
    g_peak = g[min1_idx:min2_idx]
 
    auc = np.trapz(g_peak[g_peak>1] - 1, r_peak[g_peak>1])

    return auc

def _pairing_func(x, a, b, c):
    """exponential function for fitting AUC data"""
    y = a * np.exp(-(b * x)**c)
    #y = a * np.exp(-(x/b)**c)

    return y

def compute_fit(time, auc):
    time_interval = np.asarray(time)
    popt, pcov = curve_fit(_pairing_func, time_interval, auc)
    fit = _pairing_func(time_interval, *popt)

    A = popt[0]
    tau = 1 / popt[1]
    gamma = popt[2]

    return fit, A, tau, gamma


first_peak_auc(datas, calc_fit=True)
plot_first_peak(datas)
