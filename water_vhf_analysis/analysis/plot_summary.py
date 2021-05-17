import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os

import numpy as np
import scattering
from scattering.utils.features import find_local_maxima, find_local_minima
from scipy.signal import savgol_filter
from matplotlib.ticker import MultipleLocator
from scipy import optimize
from water_vhf_analysis.utils.utils import get_txt_file
from water_vhf_analysis.utils.plotting import get_color, make_heatmap

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
    "name": "optB88 (330 K)",
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

dftb_d3 = {
    "r": np.loadtxt(get_txt_file("dftb/nvt_total_data", "r_random.txt")),
    "t": np.loadtxt(get_txt_file("dftb/nvt_total_data", "t_random.txt")),
    "g": np.loadtxt(get_txt_file("dftb/nvt_total_data", "vhf_random.txt")),
    "name": "3obw",
}

dftb_filtered = {
    "r": dftb_d3["r"],
    "t": dftb_d3["t"],
    "g": savgol_filter(dftb_d3["g"], window_length=7, polyorder=2),
    "name": "3obw/D3 (filtered)",
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


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(((x - mean) / 4 / stddev) ** 2))


datas = [IXS, spce, tip3p_ew, bk3, reaxff, dftb_d3, aimd, aimd_330]
comp_datas = [IXS, aimd, aimd_330]


def second_peak(datas, normalize=False, save=True):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(8, 8))

    for data in datas:
        print(data["name"])
        # Find nearest value to 0.1
        if normalize:
            norm_idx = find_nearest(data["t"], 0.25)
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            local_maximas = find_local_maxima(data["r"], frame, r_guess=0.44)
            maxs[i] = local_maximas[1]

        if normalize:
            if data["name"] == "IXS":
                ax.plot(
                    data["t"],
                    (maxs - 1) / (maxs[0] - 1),
                    ".",
                    lw=2,
                    label=data["name"],
                    color=get_color(data["name"]),
                )
            else:
                ax.plot(
                    data["t"],
                    (maxs - 1) / (maxs[0] - 1),
                    ls="--",
                    lw=2,
                    label=data["name"],
                    color=get_color(data["name"]),
                )
        else:
            if data["name"] == "IXS":
                ax.semilogy(
                    data["t"],
                    maxs - 1,
                    ".",
                    lw=2,
                    label=data["name"],
                    color=get_color(data["name"]),
                )
            else:
                ax.semilogy(
                    data["t"],
                    maxs - 1,
                    ls="--",
                    lw=2,
                    label=data["name"],
                    color=get_color(data["name"]),
                )

    ax.tick_params(axis="both", labelsize=labelsize)

    plt.legend(
        bbox_to_anchor=(0.5, 1.25), loc="upper center", prop={"size": fontsize}, ncol=4
    )
    plt.tight_layout()
    if normalize:
        ax.set_xlim((0.005, 0.8))
        ax.set_ylim((0.0, 1.5))
        ax.set_ylabel(r"$G_2(t) / G_2(0)$, normalized", fontsize=fontsize)
        ax.set_xlabel(r"Time, $t$ / $t(0)$, $ps$", fontsize=fontsize)
        if save:
            fig.savefig(
                "figures/overall_second_peak_normalize.png", dpi=500, bbox_inches="tight"
            )
            fig.savefig(
                "figures/overall_second_peak_normalize.pdf", dpi=500, bbox_inches="tight"
            )
    else:
        ax.set_xlim((0.005, 0.8))
        ax.set_ylim((0.01, 0.5))
        ax.set_ylabel(r"$G_2(t)-1$", fontsize=fontsize)
        ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
        if save:
            fig.savefig("figures/overall_second_peak.png", dpi=500, bbox_inches="tight")
            fig.savefig("figures/overall_second_peak.pdf", dpi=500, bbox_inches="tight")


def plot_total_subplots(datas, save=True):
    fontsize = 16

    fig = plt.figure(figsize=(20, 14))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    axes = list()
    cmap = matplotlib.cm.get_cmap("copper")
    for i in range(1, 9):
        ax = fig.add_subplot(4, 4, i)
        data = datas[i - 1]
        for idx, frame in enumerate(range(len(data["t"]))):
            if data["name"] in ["IXS"]:
                if idx % 3 != 0:
                    continue
            elif data["name"] == "CHON-2017_weak":
                if idx % 5 != 0:
                    continue
            else:
                if idx % 10 != 0:
                    continue
            ax.plot(
                data["r"], data["g"][frame], c=cmap(data["t"][frame] / data["t"][-1])
            )

        norm = matplotlib.colors.Normalize(vmin=data["t"][0], vmax=data["t"][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # cbar = plt.colorbar(sm)
        # cbar.set_label(r'Time, ps', rotation=90)
        ax.plot(data["r"], np.ones(len(data["r"])), "k--", alpha=0.6)
        ax.set_title(data["name"], fontsize=fontsize, y=1.05)

        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((round(data["r"][0], 1), round(data["r"][-1], 1)))
        ax.set_ylim((0, 3.5))

        ax.set_xlim((0, 0.8))
        xlabel = r"r, $nm$"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(r"$G(r, t)$", fontsize=fontsize)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        axes.append(ax)
    cbar = fig.colorbar(sm, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"Time, $t$, $ps$", rotation=90, fontsize=fontsize)

    axes = list()
    for i in range(9, 17):
        ax = fig.add_subplot(4, 4, i)
        data = datas[(i - 8) - 1]
        heatmap = make_heatmap(data, ax, fontsize=fontsize)
        axes.append(ax)

    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$G(r, t) - 1$", rotation=90, fontsize=fontsize)
    if save:
        plt.savefig("figures/total_subplot.png", bbox_inches="tight", dpi=500)
        plt.savefig("figures/total_subplot.pdf", bbox_inches="tight", dpi=500)


def plot_self_subplots(datas, save=True):
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    cmap = matplotlib.cm.get_cmap("copper")
    for i in range(1, 8):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i - 1]
        cutoff_max = np.where(np.isclose(data["t"], 1.0, 0.02))[0][0]
        cutoff_min = np.where(np.isclose(data["t"], 0.1, 0.05))[0][0]
        for idx, (frame, g_r) in enumerate(
            zip(data["t"][cutoff_min:cutoff_max], data["g"][cutoff_min:cutoff_max])
        ):
            if data["name"] == "IXS":
                pass
            # else:
            #    if data['t'][frame] < 0.2:
            #        continue
            if data["name"] == "optB88_filtered":
                if idx % 5 != 0:
                    continue
            ax.plot(data["r"], g_r, c=cmap(frame / data["t"][cutoff_max]))

        norm = matplotlib.colors.Normalize(
            vmin=data["t"][cutoff_min], vmax=data["t"][cutoff_max]
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data["r"], np.ones(len(data["r"])), "k--", alpha=0.6)
        ax.set_title(data["name"], fontsize=12)

        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((0, 0.2))
        ax.set_ylim((0, 200.0))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        xlabel = r"r, $nm$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$G(r, t)$")
        axes.append(ax)
    # fig.subplots_adjust(right=0.8)
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r"Time, $t$, $ps$", rotation=90, fontsize=14)
    if save:
        plt.savefig("figures/self_subplot.png", dpi=500)
        plt.savefig("figures/self_subplot.pdf", dpi=500)


def plot_heatmap(datas):
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.8)
    axes = list()
    for i in range(1, 9):
        ax = fig.add_subplot(2, 4, i)
        data = datas[i - 1]
        heatmap = make_heatmap(data, ax)
        axes.append(ax)

    plt.tight_layout()
    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.set_label(r"$G(r, t) - 1$", rotation=90, fontsize=14)
    plt.savefig("figures/heatmap.png", dpi=500)
    plt.savefig("figures/heatmap.pdf", dpi=500)


def plot_decay_subplot(datas):
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
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data["r"], frame, r_guess=0.26)[1]
            if data["name"] == "SPC/E":
                max_r.append(find_local_maxima(data["r"], frame, r_guess=0.26)[0])
        if data["name"] == "IXS":
            ax.semilogy(
                data["t"],
                maxs - 1,
                ".",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        # Grabbing every fifth data point of AIMD data
        elif data["name"] in ("optB88 (filtered)", "optB88 at 330K (filtered)", "AIMD"):
            ax.semilogy(
                data["t"][::5],
                (maxs - 1)[::5],
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs - 1,
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
    ax.set_xlim((0.00, 0.6))
    ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r"$G_1(t)-1$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.vlines(x=0.1, ymin=2e-2, ymax=2.5, color="k", ls="--")
    ax.tick_params(axis="both", labelsize=labelsize)
    fig.legend(
        bbox_to_anchor=(0.45, 1.15), loc="upper center", prop={"size": fontsize}, ncol=4
    )

    # Plot second peak decay
    ax = axes[1]
    ax.text(-0.10, 0.90, "b)", transform=ax.transAxes, size=20, weight="bold")
    for data in datas:
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
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        elif data["name"] in ("optB88 (filtered)", "optB88 at 330K (filtered)", "AIMD"):
            ax.semilogy(
                data["t"][::5],
                (maxs - 1)[::5],
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs - 1,
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.set_xlim((0.00, 0.8))
    # ax.set_ylim((.003, .5))
    ax.set_ylim((0.01, 0.5))
    ax.set_ylabel(r"$G_2(t)-1$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=labelsize)

    fig.savefig("figures/peak_decay.png", dpi=500, bbox_inches="tight")
    fig.savefig("figures/peak_decay.pdf", dpi=500, bbox_inches="tight")


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
    ax.text(-0.10, 1.0, "a)", transform=ax.transAxes, size=20, weight="bold")
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
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs - 1,
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.tick_params(axis="both", labelsize=labelsize)
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((0.01, 0.5))
    ax.set_ylabel(r"$G_2(t)-1$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    fig.legend(
        bbox_to_anchor=(0.45, 1.15), loc="upper center", prop={"size": fontsize}, ncol=4
    )

    ax = axes[1]
    ax.text(-0.10, 1.0, "b)", transform=ax.transAxes, size=20, weight="bold")
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
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            # ax.plot(data['t'], (maxs-1)/(maxs[0]-1), ls='--', lw=2, label=data['name'], color=get_color(data['name']))
            ax.plot(
                data["t"],
                min_max,
                ls="--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
    ax.set_xlim((0.005, 0.8))
    ax.set_ylim((0.0, 1.10))
    # ax.set_ylabel(r'$g_2(t) / g_2(0)$, normalized', fontsize=fontsize)
    ax.set_ylabel(r"$G_2(t)-1$, normalized", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    fig.savefig("figures/second_subplot.png", dpi=500, bbox_inches="tight")
    fig.savefig("figures/second_subplot.pdf", dpi=500, bbox_inches="tight")

def plot_total_comparison(datas, save=True):
    """Plot comparison between IXS and one or two other model"""
    fontsize = 16

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(hspace=0.8, wspace=0.7)
    axes = list()
    cmap = matplotlib.cm.get_cmap("copper")
    for i in range(1, len(datas)+1):
        ax = fig.add_subplot(2, len(datas), i)
        data = datas[i - 1]
        for idx, frame in enumerate(range(len(data["t"]))):
            if data["name"] in ["IXS"]:
                if idx % 3 != 0:
                    continue
            elif data["name"] == "CHON-2017_weak":
                if idx % 5 != 0:
                    continue
            else:
                if idx % 10 != 0:
                    continue
            ax.plot(
                data["r"], data["g"][frame], c=cmap(data["t"][frame] / data["t"][-1])
            )

        norm = matplotlib.colors.Normalize(vmin=data["t"][0], vmax=data["t"][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data["r"], np.ones(len(data["r"])), "k--", alpha=0.6)
        ax.set_title(data["name"], fontsize=fontsize, y=1.05)

        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlim((round(data["r"][0], 1), round(data["r"][-1], 1)))
        ax.set_ylim((0, 3.5))

        ax.set_xlim((0, 0.8))
        xlabel = r"r, $nm$"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(r"$G(r, t)$", fontsize=fontsize)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        axes.append(ax)
    cbar = fig.colorbar(sm, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"Time, $t$, $ps$", rotation=90, fontsize=fontsize)

    axes = list()
    for i in range(len(datas)+1, len(datas)*2+1):
        ax = fig.add_subplot(2, len(datas), i)
        data = datas[(i - len(datas)) - 1]
        heatmap = make_heatmap(data, ax, fontsize=fontsize)
        axes.append(ax)

    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$G(r, t) - 1$", rotation=90, fontsize=fontsize)
    if save:
        plt.savefig("figures/comp_subplot.png", bbox_inches="tight", dpi=500)
        plt.savefig("figures/comp_subplot.pdf", bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    #plot_total_subplots(datas)
    # plot_self_subplots(datas)
    # plot_heatmap(datas)
    # plot_decay_subplot(datas)
    # plot_second_subplot(datas)
    plot_total_comparison(comp_datas)
