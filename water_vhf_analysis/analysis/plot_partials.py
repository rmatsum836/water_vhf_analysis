import numpy as np
import mdtraj as md
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scattering
import seaborn as sns
from scattering.utils.features import find_local_maxima, find_local_minima
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from water_vhf_analysis.utils.utils import get_txt_file
from water_vhf_analysis.utils.plotting import get_color, make_heatmap

pairs = ["O_H", "O_O", "H_H"]


def get_data(pair):
    """Get data based on pair"""
    aimd = {
        "r": np.loadtxt(get_txt_file("aimd/partial_overlap_nvt", f"r_final_{pair}.txt")),
        "t": np.loadtxt(get_txt_file("aimd/partial_overlap_nvt", f"t_final_{pair}.txt")),
        "g": np.loadtxt(
            get_txt_file("aimd/partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "optB88 (AIMD)",
    }

    spce = {
        "r": np.loadtxt(get_txt_file("spce/partial_overlap_nvt", f"r_final_{pair}.txt")),
        "t": np.loadtxt(get_txt_file("spce/partial_overlap_nvt", f"t_final_{pair}.txt")),
        "g": np.loadtxt(
            get_txt_file("spce/partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "SPC/E (CMD)",
    }

    tip3p_ew = {
        "r": np.loadtxt(
            get_txt_file("tip3p_ew/partial_overlap_nvt", f"r_final_{pair}.txt")
        ),
        "t": np.loadtxt(
            get_txt_file("tip3p_ew/partial_overlap_nvt", f"t_final_{pair}.txt")
        ),
        "g": np.loadtxt(
            get_txt_file("tip3p_ew/partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "TIP3P_EW (CMD)",
    }

    bk3 = {
        "r": np.loadtxt(get_txt_file("bk3/partial_overlap_nvt", f"r_final_{pair}.txt")),
        "t": np.loadtxt(get_txt_file("bk3/partial_overlap_nvt", f"t_final_{pair}.txt")),
        "g": np.loadtxt(get_txt_file("bk3/partial_overlap_nvt", f"vhf_final_{pair}.txt")),
        "name": "BK3 (Polarizable CMD)",
    }

    reaxff = {
        "r": np.loadtxt(
            get_txt_file("reaxff/9000_partial_overlap_nvt", f"r_final_{pair}.txt")
        ),
        "t": np.loadtxt(
            get_txt_file("reaxff/9000_partial_overlap_nvt", f"t_final_{pair}.txt")
        )-200,
        "g": np.loadtxt(
            get_txt_file("reaxff/9000_partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "CHON-2017_weak (ReaxFF)",
    }

    aimd_330 = {
        "r": np.loadtxt(
            get_txt_file("aimd/330k/partial_overlap_nvt", f"r_final_{pair}.txt")
        ),
        "t": np.loadtxt(
            get_txt_file("aimd/330k/partial_overlap_nvt", f"t_final_{pair}.txt")
        ),
        "g": np.loadtxt(
            get_txt_file("aimd/330k/partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "optB88 at 330 K (AIMD)",
    }

    aimd_filtered_330 = {
        "r": aimd_330["r"],
        "t": aimd_330["t"],
        "g": savgol_filter(aimd_330["g"], window_length=11, polyorder=4),
        "name": "optB88 at 330K (filtered)",
    }

    aimd_filtered = {
        "r": aimd["r"],
        "t": aimd["t"],
        "g": savgol_filter(aimd["g"], window_length=11, polyorder=4),
        "name": "optB88 (filtered)",
    }

    dftb = {
        "r": np.loadtxt(get_txt_file("dftb/partial_overlap_nvt", f"r_final_{pair}.txt")),
        "t": np.loadtxt(get_txt_file("dftb/partial_overlap_nvt", f"t_final_{pair}.txt")),
        "g": np.loadtxt(
            get_txt_file("dftb/partial_overlap_nvt", f"vhf_final_{pair}.txt")
        ),
        "name": "3obw (DFTB)",
    }

    datas = [spce, tip3p_ew, bk3, reaxff, dftb, aimd, aimd_330]

    return datas


def plot_peak_subplots(save=True):
    fontsize = 18
    labelsize = 20
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)

    # Plot OH peak decay
    peak_guess = 0.18
    ylim = (0.5, 10)
    ax = axes[0][0]
    ax.text(-0.10, 1.05, "a)", transform=ax.transAxes, size=20, weight="bold")
    datas = get_data("O_H")
    colors = sns.color_palette("muted", len(datas))
    for data in datas:
        r_low = np.where(data["r"] > 0.16)[0][0]
        r_high = np.where(data["r"] < 0.2)[0][-1]
        r_range = data["r"][r_low:r_high]
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"][:50]):
            g_range = frame[r_low:r_high]
            max_r, max_g = find_local_maxima(r_range, g_range, r_guess=peak_guess)
            maxs[i] = max_g
        ax.semilogy(
            data["t"],
            maxs,
            "--",
            lw=3,
            label=data["name"],
            color=get_color(data["name"]),
        )

    #ax.set_xlim((0.00, 0.21))
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.set_xlim((0.00, 0.12))
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$G_{\mathrm{OH}_1}(t)$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    # Plot normalized OH peak decay
    peak_guess = 0.18
    ylim = (0.00, 1.15)
    shift = True
    ax = axes[0][1]
    ax.text(-0.10, 1.05, "b)", transform=ax.transAxes, size=20, weight="bold")
    datas = get_data("O_H")
    for data in datas:
        r_low = np.where(data["r"] > 0.16)[0][0]
        r_high = np.where(data["r"] < 0.2)[0][-1]
        r_range = data["r"][r_low:r_high]
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"][:50]):
            g_range = frame[r_low:r_high]
            max_r, max_g = find_local_maxima(r_range, g_range, r_guess=peak_guess)
            maxs[i] = max_g
        min_max = ((maxs - 1) - np.min(maxs - 1)) / (
            np.max(maxs - 1) - np.min(maxs - 1)
        )
        ax.plot(
            data["t"],
            min_max,
            "--",
            lw=3,
            label=data["name"],
            color=get_color(data["name"]),
        )

    #ax.set_xlim((0.00, 0.21))
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.set_xlim((0.00, 0.12))
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$G_{\mathrm{OH}_1}(t)-1$, normalized", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    # Plot HH peak decay
    peak_guess = 0.23
    ylim = (0.001, 2)
    shift = False
    ax = axes[1][0]
    ax.text(-0.10, 1.05, "c)", transform=ax.transAxes, size=20, weight="bold")
    datas = get_data("H_H")
    for data in datas:
        r_low = np.where(data["r"] > 0.16)[0][0]
        r_high = np.where(data["r"] < 0.25)[0][-1]
        r_range = data["r"][r_low:r_high]
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            g_range = frame[r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g_range, r_guess=peak_guess)
            # print(r_max)
            maxs[i] = g_max
        if shift == True:
            ax.semilogy(
                data["t"],
                maxs - 1,
                "--",
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs,
                "--",
                lw=3,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.set_xlim((0.00, 0.8))
    # ax.set_ylim(ylim)
    if shift == True:
        ax.set_ylabel(r"$G_{\mathrm{HH}_1}(t)-1$", fontsize=fontsize)
    else:
        ax.set_ylabel(r"$G_{\mathrm{HH}_1}(t)$", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    # Plot normalized HH peak decay
    peak_guess = 0.23
    ylim = (0, 1.05)
    ax = axes[1][1]
    legend_ax = ax
    ax.text(-0.10, 1.05, "d)", transform=ax.transAxes, size=20, weight="bold")
    datas = get_data("H_H")
    for data in datas:
        r_low = np.where(data["r"] > 0.16)[0][0]
        r_high = np.where(data["r"] < 0.25)[0][-1]
        r_range = data["r"][r_low:r_high]
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            g_range = frame[r_low:r_high]
            r_max, g_max = find_local_maxima(r_range, g_range, r_guess=peak_guess)
            # print(r_max)
            maxs[i] = g_max
        min_max = ((maxs - 1) - np.min(maxs - 1)) / (
            np.max(maxs - 1) - np.min(maxs - 1)
        )
        ax.plot(
            data["t"],
            min_max,
            "--",
            lw=3,
            label=data["name"],
            color=get_color(data["name"]),
        )

    ax.set_xlim((0.00, 0.8))
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$G_{\mathrm{HH}_1}(t)-1$, normalized", fontsize=fontsize)
    ax.set_xlabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)

    handles, labels = legend_ax.get_legend_handles_labels()
    lgd = fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 1.12),
        fontsize=fontsize,
        loc="upper center",
        ncol=3,
    )

    plt.tight_layout()
    if save:
        fig.savefig("figures/partial_peak_decay.png", dpi=500, bbox_inches="tight")
        fig.savefig("figures/partial_peak_decay.pdf", dpi=500, bbox_inches="tight")


def plot_oh_peak(datas, filename, ylim=(0, 3), plot_max=False):
    """Plot H-bond peak around 0.18 nm

    datas : list of dicts
        List of VHF data
    filename : str
        File save data out to
    ylim : tuple
        ylim for Matplotlib
    plot_max : bool, default=False
        Plot local maxima for peaks
    """
    fontsize = 12
    labelsize = 14
    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    axes = list()
    cmap = matplotlib.cm.get_cmap("copper")
    for i in range(1, len(datas) + 1):
        ax = fig.add_subplot(4, 4, i)
        data = datas[i - 1]

        # Test to check where point is
        r_low = np.where(data["r"] > 0.16)[0][0]
        r_high = np.where(data["r"] < 0.2)[0][-1]
        r_range = data["r"][r_low:r_high]

        if data["name"] == "CHON-2017_weak (ReaxFF)":
            for frame in range(len(data["t"][:25])):
                g_range = data["g"][frame][r_low:r_high]
                max_r, max_g = find_local_maxima(r_range, g_range, r_guess=0.18)
                ax.plot(
                    data["r"],
                    data["g"][frame],
                    c=cmap(data["t"][:25][frame] / data["t"][:25][-1]),
                )
                if plot_max:
                    ax.scatter(max_r, max_g, c="k")
                ax.set_title(data["name"], fontsize=fontsize)
            norm = matplotlib.colors.Normalize(
                vmin=data["t"][0], vmax=data["t"][:25][-1]
            )
        else:
            t_max = 50
            for frame in range(0, len(data["t"][:t_max])):
                # Test to check where point is
                g_range = data["g"][frame][r_low:r_high]
                max_r, max_g = find_local_maxima(r_range, g_range, r_guess=0.18)

                ax.plot(
                    data["r"],
                    data["g"][frame],
                    c=cmap(data["t"][:t_max][frame] / data["t"][:t_max][-1]),
                )
                if plot_max:
                    ax.scatter(max_r, max_g, c="k")
                ax.set_title(data["name"], fontsize=fontsize, y=1.05)
            norm = matplotlib.colors.Normalize(
                vmin=data["t"][0], vmax=data["t"][:t_max][-1]
            )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.plot(data["r"], np.ones(len(data["r"])), "k--", alpha=0.6)

        # ax.set_xlim((0, 0.8))
        ax.set_xlim((0.1, 0.23))
        ax.set_ylim(ylim)
        xlabel = r"r, $nm$"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(r"$G(r, t)$", fontsize=fontsize)
        ax.tick_params(labelsize=14)
        axes.append(ax)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label(r"Time, $t$, $ps$", rotation=90, fontsize=fontsize)
    plt.savefig(filename, bbox_inches="tight", dpi=500)


# Plot all in one subplot
def plot_vhf_subplots(datas, filename=None, ylim=(0, 3)):
    fontsize = 16
    labelsize = 16
    fig = plt.figure(figsize=(20, 14))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    axes = list()
    cmap = matplotlib.cm.get_cmap("copper")
    for i in range(1, len(datas) + 1):
        ax = fig.add_subplot(4, 4, i)
        data = datas[i - 1]
        for idx, frame in enumerate(range(len(data["t"]))):
            if data["name"] == "CHON-2017_weak (ReaxFF)":
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

        ax.set_xlim((0, 0.8))
        ax.set_ylim(ylim)
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
    for i in range(9, 16):
        ax = fig.add_subplot(4, 4, i)
        data = datas[(i - 7) - 2]
        heatmap = make_heatmap(data, ax, fontsize=fontsize)
        axes.append(ax)

    cbar = fig.colorbar(heatmap, ax=axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$G(r, t) - 1$", rotation=90, fontsize=fontsize)
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=500)


def first_peak_height(datas, filename, peak_guess=0.3, ylim=((0.06, 0.18)), shift=True):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))

    for data in datas:
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data["r"], frame, r_guess=peak_guess)[1]
        if shift == True:
            ax.semilogy(
                data["t"],
                maxs - 1,
                "--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )
        else:
            ax.semilogy(
                data["t"],
                maxs,
                "--",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.set_xlim((0.01, 0.8))
    ax.set_ylim(ylim)
    if shift == True:
        ax.set_ylabel(r"$G_1(t)-1$", fontsize=fontsize)
    else:
        ax.set_ylabel(r"$G_1(t)$", fontsize=fontsize)
    ax.set_xlabel("Time (ps)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": fontsize})
    plt.tight_layout()
    fig.savefig(filename, dpi=500)


def first_oh_peak(datas, filename, peak_guess=0.18, ylim=((0.6, 1.8))):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))

    for data in datas:
        maxs = np.zeros(len(data["t"]))
        if data["name"] == "optB88_filtered":
            for i, frame in enumerate(data["g"][:1000]):
                maxs[i] = find_local_maxima(data["r"], frame, r_guess=peak_guess)[1]
        else:
            for i, frame in enumerate(data["g"][:50]):
                maxs[i] = find_local_maxima(data["r"], frame, r_guess=peak_guess)[1]
        ax.semilogy(
            data["t"],
            maxs,
            "--",
            lw=2,
            label=data["name"],
            color=get_color(data["name"]),
        )

    ax.set_xlim((0.01, 0.11))
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$G_1(t)$", fontsize=fontsize)
    ax.set_xlabel("Time (ps)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": fontsize})
    plt.tight_layout()
    fig.savefig(filename, dpi=500)


def first_peak_min(datas, filename, peak_guess=0.3):
    fontsize = 14
    labelsize = 14
    fig, ax = plt.subplots(figsize=(9, 5))

    for data in datas:
        mins = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                mins[i] = np.nan
                continue
            mins[i] = find_local_minima(data["r"], frame, r_guess=peak_guess)[1]
        print(data["name"])
        ax.semilogy(
            data["t"],
            mins,
            "--",
            lw=2,
            label=data["name"],
            color=get_color(data["name"]),
        )

    # ax.set_xlim((0.01, 0.6))
    # ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r"$G_{1min}(t)-1$", fontsize=fontsize)
    ax.set_xlabel("Time (ps)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=labelsize)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": fontsize})

    plt.tight_layout()
    fig.savefig(filename, dpi=500)


def first_second_peak(datas, filename, first_peak_guess=0.25, second_peak_guess=0.4):
    fig, ax = plt.subplots(figsize=(8, 5))
    # First peak
    for data in datas:
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data["r"], frame, r_guess=first_peak_guess)[1]
        ax.semilogy(
            data["t"],
            maxs,
            "--",
            lw=2,
            label=data["name"],
            color=get_color(data["name"]),
        )
    # Second peak
    for data in datas:
        maxs = np.zeros(len(data["t"]))
        for i, frame in enumerate(data["g"]):
            if data["t"][i] < 0.0:
                maxs[i] = np.nan
                continue
            maxs[i] = find_local_maxima(data["r"], frame, r_guess=second_peak_guess)[1]
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
                ".-",
                lw=2,
                label=data["name"],
                color=get_color(data["name"]),
            )

    ax.set_xlim((0.01, 0.6))
    # ax.set_ylim((3e-2, 2.5))
    ax.set_ylabel(r"Peak height")
    ax.set_xlabel(r"Time, t, $ps$")

    # plt.legend()
    fig.savefig(filename, dpi=500)


if __name__ == "__main__":
    for pair in pairs:
        if pair == "O_H":
            ylim = (0, 2)
            peak_guess = 0.3
        elif pair == "O_O":
            ylim = (0, 3.5)
        elif pair == "H_H":
            ylim = (0, 2)
            peak_guess = 0.23

        datas = get_data(pair)

        if pair == "O_H":
            plot_oh_peak(datas, ylim=ylim, filename=f"figures/O_H_hbond_peak.png")
            plot_oh_peak(datas, ylim=ylim, filename=f"figures/O_H_hbond_peak.pdf")
            # first_oh_peak(datas,filename=f'figures/{pair}_first_peak.png')
            # first_oh_peak(datas,filename=f'figures/{pair}_first_peak.pdf')

        plot_vhf_subplots(datas, ylim=ylim, filename=f"figures/{pair}_subplot.pdf")
        plot_vhf_subplots(datas, ylim=ylim, filename=f"figures/{pair}_subplot.png")

        # if pair == 'H_H':
        #    first_peak_height(datas, peak_guess=peak_guess, ylim=(0.8, 1.8), filename=f'figures/{pair}_first_peak.pdf', shift=False)
        #    first_peak_height(datas, peak_guess=peak_guess, ylim=(0.8, 1.8), filename=f'figures/{pair}_first_peak.png', shift=False)
        # if pair == 'H_H':
        #    first_second_peak(datas, first_peak_guess=0.25, second_peak_guess=0.4, filename=f'figures/{pair}_first_second.pdf')
        #    first_second_peak(datas, first_peak_guess=0.25, second_peak_guess=0.4, filename=f'figures/{pair}_first_second.png')
        # if pair == 'O_H':
        #    first_peak_min(datas, peak_guess=0.2, filename=f'figures/{pair}_minima.pdf')
        #    first_peak_min(datas, peak_guess=0.2, filename=f'figures/{pair}_minima.png')
        # if pair == 'O_O':
        #    first_peak_height(datas, peak_guess=0.3, ylim=(0.01, 2.5), filename=f'figures/{pair}_first_peak.pdf')
        #    first_peak_height(datas, peak_guess=0.3, ylim=(0.01, 2.5), filename=f'figures/{pair}_first_peak.png')

    plot_peak_subplots()
