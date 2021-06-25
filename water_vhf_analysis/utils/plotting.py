import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def get_color(name):
    color_dict = dict()
    color_list = [
        "TIP3P_EW (CMD)",
        "placeholder",
        "SPC/E (CMD)",
        "CHON-2017_weak (ReaxFF)",
        "BK3 (Polarizable CMD)",
        "3obw (DFTB)",
        "optB88 (filtered)",
        "optB88 (AIMD)",
        "optB88 at 330 K (AIMD)",
    ]
    colors = sns.color_palette("muted", len(color_list))
    for model, color in zip(color_list, colors):
        color_dict[model] = color

    color_dict["IXS"] = "black"

    return color_dict[name]


def make_heatmap(data, ax, color_t=None, v=0.1, fontsize=14):
    """
    Plot heatmap for g_r_t

    Parameters
    ----------
    data : dict
        VHF data in dict format
    color_t : np.array, default=None
        Array of time data from simulation.  Required with IXS data due to different time intervals from 
        simulation data, used for uniformity.
    ax : mpl.plt.ax
        Axis to plot heatmap to
    v : float
        Data range to cover
    fontsize : int
        Fontsize of plot

    Returns
    -------
    """
    if color_t is None:
        color_t = data["t"]
    heatmap = ax.imshow(
        data["g"] - 1,
        vmin=-v,
        vmax=v,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=(data["r"][0], data["r"][-1], color_t[0], color_t[-1]),
    )
    ax.grid(False)
    ax.set_xlim((round(data["r"][0], 1), 0.8))
    ax.set_ylim(
        (0, 1)
    )  # ax.set_ylim((round(data['t'][0], 1), round(data['t'][-1], 2)))

    xlabel = r"r, $nm$"
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(r"Time, $t$, $ps$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(data["name"], fontsize=fontsize, y=1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))

    return heatmap
