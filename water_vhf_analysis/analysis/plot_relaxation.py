import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from water_vhf_analysis.utils.utils import get_txt_file
from water_vhf_analysis.utils.plotting import get_color
from water_vhf_analysis.utils.analysis import get_auc, compute_fit


aimd = {
    "r": np.loadtxt(get_txt_file("aimd/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("aimd/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("aimd/nvt_total_data", "vhf_random.txt")),
    "name": "optB88",
}

aimd_330 = {
    "r": np.loadtxt(get_txt_file("aimd/330k/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("aimd/330k/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("aimd/330k/nvt_total_data", "vhf_random.txt")),
    "name": "optB88 (330K)",
}

aimd_filtered = {
    "r": aimd["r"],
    "t": aimd["t"],
    "g": savgol_filter(aimd["g"], window_length=11, polyorder=4),
    "name": "optB88 (filtered)",
}

aimd_filtered_330 = {
    "r": aimd_330["r"],
    "t": aimd_330["t"],
    "g": savgol_filter(aimd_330["g"], window_length=7, polyorder=3),
    "name": "optB88 at 330K (filtered)",
}

bk3 = {
    "r": np.loadtxt(get_txt_file("bk3/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("bk3/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("bk3/nvt_total_data", "vhf_random.txt")),
    "name": "BK3",
}

dftb = {
    "r": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "vhf_random.txt")),
    "name": "DFTB_noD3/3obw",
}

dftb_d3 = {
    "r": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("dftb/nvt_total_data/2ns", "vhf_random.txt")),
    "name": "DFTB_D3/3obw",
}

dftb_filtered = {
    "r": dftb_d3["r"],
    "t": dftb_d3["t"],
    "g": savgol_filter(dftb_d3["g"], window_length=7, polyorder=2),
    "name": "DFTB_D3 (filtered)",
}

spce = {
    "r": np.loadtxt(get_txt_file("spce/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("spce/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("spce/nvt_total_data", "vhf_random.txt")),
    "name": "SPC/E",
}

tip3p_ew = {
    "r": np.loadtxt(get_txt_file("tip3p_ew/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("tip3p_ew/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("tip3p_ew/nvt_total_data", "vhf_random.txt")),
    "name": "TIP3P_EW",
}

reaxff = {
    "r": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "vhf_random.txt")),
    "name": "CHON-2017_weak",
}

IXS = {
    "name": "IXS",
    "r": 0.1 * np.loadtxt(get_txt_file("expt", "R_1811pure.txt"))[0],
    "t": np.loadtxt(get_txt_file("expt", "t_1811pure.txt"))[:, 0],
    "g": 1 + np.loadtxt(get_txt_file("expt", "VHF_1811pure.txt")),
}

datas = [IXS, spce, tip3p_ew, bk3, reaxff, dftb_d3, aimd, aimd_330]


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
        r = data["r"]

        # Get the peak height for the first peak
        # `r_low` and `r_high` are attempt to add bounds for the first peak
        r_low = np.where(r > 0.26)[0][0]
        r_high = np.where(r < 0.34)[0][-1]
        r_range = r[r_low:r_high]
        t = data["t"]
        I = np.empty_like(t)
        I[:] = np.nan
        # Loop through times
        for i in range(0, t.shape[0], 5):
            g = data["g"][i][r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g, 0.28)
            print(f"R at time {t[i]} is: {r_max}")

            plt.scatter(
                data["t"][i], r_max, color=get_color(data["name"]), label=data["name"]
            )

        # Get the peak height for the second peak
        # `r_low` and `r_high` are attempt to add bounds for the first peak
        r_low = np.where(r > 0.4)[0][0]
        r_high = np.where(r < 0.55)[0][-1]
        r_range = r[r_low:r_high]
        t = data["t"]
        I = np.empty_like(t)
        I[:] = np.nan
        # Loop through times
        for i in range(0, t.shape[0], 5):
            g = data["g"][i][r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g, 0.45)

            plt.scatter(
                data["t"][i], r_max, color=get_color(data["name"]), label=data["name"]
            )

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    plt.ylabel("Peak Position (nm)")
    plt.xlabel("t (ps)")
    plt.ylim((0.25, 0.5))
    plt.xlim((0.0, 2.0))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
        prop={"size": 12},
        ncol=4,
    )
    plt.savefig("figures/peak_locations.pdf", dpi=500, bbox_inches="tight")
    plt.savefig("figures/peak_locations.png", dpi=500, bbox_inches="tight")


def first_peak_auc(datas, si=False):
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
    columns = ("A_1", "tau_1", "gamma_1", "A_2", "tau_2", "gamma_2")
    index = [i["name"] for i in datas]
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(1, len(datas) + 1):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i - 1]
        r = data["r"] * 10  # convert from angstroms to nm
        t = data["t"]
        g = data["g"]

        I = np.empty_like(t)
        I[:] = np.nan

        # Get area under the curve
        for i in range(0, t.shape[0]):
            I[i] = get_auc(data, i)
        ls = "--"

        ax.semilogy(t[::2], I[::2], ls=ls, lw=2, label=data["name"])

        # Get finite values
        I_idx = np.where(~np.isnan(I))
        I = I[np.isfinite(I)]
        t = t[I_idx]

        # upper_limit = np.where(t < 0.28)[0][-1]
        if si:
            upper_limit = np.where(t < 0.7)[0][-1]
            t = t[:upper_limit]
            I = I[:upper_limit]
        else:
            if data["name"] not in ("IXS"):
                if data["name"] == "optB88":
                    upper_limit = np.where(t < 1.15)[0][-1]
                elif data["name"] == "CHON-2017_weak":
                    upper_limit = np.where(t < 0.85)[0][-1]
                elif data["name"] == "DFTB_D3/3obw":
                    upper_limit = np.where(t < 0.7)[0][-1]
                else:
                    upper_limit = np.where(t < 1.00)[0][-1]
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
            # df.loc[data["name"]]["A_1"] = popt[0]
            # df.loc[data["name"]]["tau_1"] = 1 / popt[1]
            # df.loc[data["name"]]["A_2"] = popt[2]
            # df.loc[data["name"]]["tau_2"] = 1 / popt[3]
            # df.loc[data["name"]]["gamma_2"] = popt[4]
            # print(data["name"])
            # print(f"tau_1 is: {1/popt[1]}")
            # print(f"A_1 is: {popt[0]}")
            # print("gamma_1 is: 1.57")
            # print(f"tau_2 is: {1/popt[3]}")
            # print(f"A_2 is: {popt[2]}")
            # print(f"gamma_2 is: {popt[4]}")
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
            # df.loc[data["name"]]["A_1"] = popt[3]
            # df.loc[data["name"]]["tau_1"] = 1 / popt[4]
            # df.loc[data["name"]]["A_2"] = popt[0]
            # df.loc[data["name"]]["tau_2"] = 1 / popt[1]
            # df.loc[data["name"]]["gamma_2"] = popt[2]
            # print(data["name"])
            # print(f"tau_1 is: {1/popt[4]}")
            # print(f"A_1 is: {popt[3]}")
            # print(f"tau_2 is: {1/popt[1]}")
            # print(f"A_2 is: {popt[0]}")
            # print(f"gamma_2 is: {popt[2]}")
        ax.semilogy(t, fit, linestyle=ls, color="k", label=f"{data['name']}_fit")
        ax.set_title(data["name"], fontsize=12)

        # Plot the compressed exponential functions given from 2018 Phys. Rev.
        # A_t = 0.42*(np.exp(-(t/0.12)**1.57)) + 0.026*(np.exp(-(t/0.4)**4.1))
        # ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 310 K")
        # A_t = 0.45*(np.exp(-(t/0.12)**1.57)) + 0.018*(np.exp(-(t/0.43)**12.8))
        # ax.plot(t, A_t, label="IXS fit (2018 Phys. Rev.) at 295 K K")
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        # ax.set_xlim((0.00, 1.5))
        ax.set_xlim((0.00, 1.0))
        ax.set_ylim((5e-3, 1.0))
        # ax.set_ylim((5e-4, 1.0))
        ax.set_ylabel(r"$A(t)$")
        ax.set_xlabel("Time (ps)")

    if si:
        plt.savefig("figures/first_peak_auc_si.pdf")
        plt.savefig("figures/first_peak_auc_si.png")
        df.to_csv("tables/first_peak_fits_si.csv")
    else:
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
    ax.text(-0.10, 0.90, "a)", transform=ax.transAxes, size=20, weight="bold")
    ax.set_prop_cycle("color", colors)
    max_r = list()
    for data in datas:
        maxs = np.zeros(len(data["t"]))
        r = data["r"]
        r_low = np.where(r > 0.15)[0][0]
        r_high = np.where(r < 0.32)[0][-1]
        r_range = r[r_low:r_high]
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            g_range = frame[r_low:r_high]
            # maxs[i] = find_local_maxima(data['r'], frame, r_guess=0.26)[1]
            maxs[i] = find_local_maxima(r_range, g_range, r_guess=0.28)[1]
            if data["name"] == "SPC/E":
                # max_r.append(find_local_maxima(data['r'], frame, r_guess=0.26)[0])
                max_r.append(find_local_maxima(r_range, g_range, r_guess=0.28)[0])
        if data["name"] == "IXS":
            ax.semilogy(
                data["t"],
                maxs - 1,
                ".",
                lw=4,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs - 1,
                ls="--",
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )
    if si:
        ax.set_xlim((0.00, 1.25))
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
    else:
        ax.set_xlim((0.00, 0.6))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r"$g_1(t)-1$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color="k", ls="--")
    ax.tick_params(axis="both", labelsize=labelsize)
    fig.legend(
        bbox_to_anchor=(0.45, 1.10), loc="upper center", prop={"size": fontsize}, ncol=4
    )

    ax = axes[1]
    ax.text(-0.10, 0.90, "b)", transform=ax.transAxes, size=20, weight="bold")
    for data in datas:
        r = data["r"] * 10  # convert from angstroms to nm
        t = data["t"]
        g = data["g"]

        I = np.empty_like(t)
        I[:] = np.nan

        # Get area under the curve
        for i in range(0, t.shape[0], 2):
            I[i] = get_auc(data, i)

        ls = "--"
        if data["name"] == "IXS":
            ax.semilogy(t, I, marker=".", label=data["name"], lw=4, color="k")
        else:
            # Get rid of NANs with [::2]
            ax.semilogy(
                t[::2],
                I[::2],
                ls=ls,
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )

        if si:
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.set_xlim((0.00, 1.25))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.set_xlim((0.00, 0.6))
        ax.set_ylim((5e-3, 1.0))
        ax.set_ylabel(r"A($t$)", fontsize=fontsize)
        ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=labelsize)
    if si:
        fig.savefig("figures/first_subplot_si.png", dpi=500, bbox_inches="tight")
        fig.savefig("figures/first_subplot_si.pdf", dpi=500, bbox_inches="tight")
    else:
        fig.savefig("figures/first_subplot.png", dpi=500, bbox_inches="tight")
        fig.savefig("figures/first_subplot.pdf", dpi=500, bbox_inches="tight")


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

        a_t = A * np.exp(-((time / tau) ** gamma))
        ax.plot(time, a_t, label=name)

        ax.set_yscale("log")

    pre = 0.448 * np.exp(-((time / 0.1235) ** 1.57))
    ax.plot(time, pre, label="IXS, 2018 Phys. Rev.")

    ax.set_ylim(5e-3, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("t/ps")
    plt.legend()
    fig.savefig("figures/tau_fit.pdf", dpi=500, bbox_inches="tight")


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
        r = data["r"]
        t = data["t"]
        g = data["g"]
        rho = 4 * np.pi * data["nwaters"] / data["volume"]  # molecules / nm^3

        I = np.empty_like(t)
        I[:] = np.nan

        # Get area under the curve
        # for i in range(0, t.shape[0], 2):
        for i in range(0, t.shape[0]):
            I[i] = get_cn(data, i) * rho
        ls = "--"

        ax.plot(t[::2], I[::2], ls=ls, lw=2, label=data["name"])

    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.set_ylabel("Coordination Number", fontsize=fontsize)
    ax.set_xlim((0, 0.6))
    fig.savefig("figures/cn_vs_t.pdf", dpi=500, bbox_inches="tight")


def plot_fits():
    """
    Plot fits of first peak, first subplot shows the first step, second subplot second step
    """
    df = pd.read_csv("tables/first_peak_fits.csv")

    # Plot first-step fit
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(df)):
        data = df.loc[i]
        time = np.arange(0, 2, 0.1)
        ax.plot(
            time,
            data["A_1"] * np.exp(-((time / data["tau_1"]) ** data["gamma_1"])),
            label=data[0],
            color=get_color(data[0]),
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$A_1(t)$")
    ax.set_ylim(5e-4, 1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Time, $t$, $ps$")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
        prop={"size": 12},
        ncol=4,
    )

    fig.savefig("figures/first_step_decay.png", dpi=500, bbox_inches="tight")

    # Plot second-step fit
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(len(df)):
        data = df.loc[i]
        fit_two = data["A_2"] * np.exp(-((time / data["tau_2"]) ** data["gamma_2"]))
        fit_norm = fit_two / fit_two[0]
        axes[0].plot(time, fit_two, label=data[0], color=get_color(data[0]))
        axes[1].plot(time, fit_norm, label=data[0], color=get_color(data[0]))
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(r"Time, $t$, $ps$")
    axes[0].set_ylabel(r"$A_2(t)$")
    axes[1].set_ylabel(r"$A_2(t) / A_2(0)$")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.05, 1.15),
        loc="upper center",
        prop={"size": 12},
        ncol=4,
    )

    fig.savefig("figures/second_step_decay.png", dpi=500, bbox_inches="tight")


def plot_second_subplot(datas):
    """ Plot the height of second peak, and normalized height of second peak"""
    fontsize = 18
    labelsize = 18
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("muted", len(datas))

    ax = axes[0]
    ax.text(-0.10, 0.95, "a)", transform=ax.transAxes, size=20, weight="bold")
    ax.set_prop_cycle("color", colors)
    for data in datas:
        print(data["name"])
        # Find nearest value to 0.1
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data["r"], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        if data["name"] == "IXS":
            ax.semilogy(
                data["t"],
                maxs - 1,
                ".",
                lw=4,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs - 1,
                ls="--",
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.tick_params(axis="both", labelsize=labelsize)
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((0.01, 0.5))
    ax.set_ylabel(r"$g_2(t)-1$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    fig.legend(
        bbox_to_anchor=(0.45, 1.10), loc="upper center", prop={"size": fontsize}, ncol=4
    )

    ax = axes[1]
    ax.text(-0.10, 0.95, "b)", transform=ax.transAxes, size=20, weight="bold")
    for data in datas:
        print(data["name"])
        # Find nearest value to 0.1
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data["r"], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        min_max = ((maxs - 1) - np.min(maxs - 1)) / (
            np.max(maxs - 1) - np.min(maxs - 1)
        )
        if data["name"] == "IXS":
            # ax.plot(data['t'], (maxs-1)/(maxs[0]-1), '.', lw=2, label=data['name'], color=get_color(data['name']))
            ax.plot(
                data["t"],
                min_max,
                ".",
                lw=4,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            # ax.plot(data['t'], (maxs-1)/(maxs[0]-1), ls='--', lw=2, label=data['name'], color=get_color(data['name']))
            ax.plot(
                data["t"],
                min_max,
                ls="--",
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((0.2, 1.10))
    # ax.set_ylabel(r'$g_2(t) / g_2(0)$, normalized', fontsize=fontsize)
    ax.set_ylabel(r"$g_2(t)-1$, normalized", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    fig.savefig("figures/second_subplot.png", dpi=500, bbox_inches="tight")
    fig.savefig("figures/second_subplot.pdf", dpi=500, bbox_inches="tight")


# plot_first_fit()
first_peak_auc(datas)
first_peak_auc(datas, si=True)
plot_peak_locations(datas)
plot_first_peak_subplot(datas, si=True)
plot_first_peak_subplot(datas)
# first_cn(datas)
plot_fits()
plot_second_subplot(datas)
