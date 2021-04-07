import numpy as np
import scipy
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scattering.utils.features import find_local_maxima, find_local_minima


def get_auc(data, idx):
    """ Get AUC of first peak"""
    r = data["r"] * 10
    g = data["g"][idx]

    min1, _ = find_local_minima(r, data["g"][idx], 0.15 * 10)
    min2, _ = find_local_minima(r, data["g"][idx], 0.34 * 10)  # Changed from 3.6 to 3.4

    # When this occurs, min2 is usually too low
    if min1 == min2:
        min2 = 0.34 * 10

    min1_idx = np.where(np.isclose(r, min1, rtol=0.02))[0][0]
    min2_idx = np.where(np.isclose(r, min2, rtol=0.02))[0][0]

    r_peak = r[min1_idx:min2_idx]
    g_peak = g[min1_idx:min2_idx]

    auc = np.trapz(g_peak[g_peak > 1] - 1, r_peak[g_peak > 1])

    return auc


def get_cn(data, idx):
    """ Get integral of g(r) of first peak for CN"""
    r = data["r"]
    g = data["g"][idx]

    # I have also tried setting the bounds to be 2.6 angstroms and 3.2 nm for all frames
    min1 = 0.24
    min2, _ = find_local_minima(r, data["g"][idx], 0.34)  # Changed from 3.6 to 3.4

    # When this occurs, min2 is usually too low
    if min1 == min2:
        min2 = 0.34

    min1_idx = np.where(np.isclose(r, min1))[0][0]
    min2_idx = np.where(np.isclose(r, min2))[0][0]

    r_peak = r[min1_idx:min2_idx]
    g_peak = g[min1_idx:min2_idx]

    auc = np.trapz(g_peak * (r_peak) ** 2, r_peak)

    return auc


def _pairing_func(x, a, b, c, d, e, f):
    """exponential function for fitting AUC data"""
    y = a * np.exp(-((b * x) ** c)) + d * np.exp(-((e * x) ** f))

    return y


def compute_fit(time, auc):
    """Compute fit of auc"""
    time_interval = np.asarray(time)
    bounds = (
        (-np.inf, 5, 0, -np.inf, 0, -np.inf),
        (np.inf, np.inf, 10, np.inf, 10, np.inf),
    )
    popt, pcov = curve_fit(
        _pairing_func, time_interval, auc, bounds=bounds, maxfev=5000
    )
    fit = _pairing_func(time_interval, *popt)

    return fit, popt
