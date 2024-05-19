from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.transforms import Bbox, ScaledTranslation

rcParams = mpl.rcParams
DEFAULT_ALPHA = 0.95
DEFAULT_LABELS = {
    "part_pt": "Particle $p_{\\mathrm{T}}$ [GeV]",
    "part_eta": "Particle $\\eta$",
    "part_phi": "Particle $\\phi$",
    "part_ptrel": "Particle $p_{\\mathrm{T}}^\\mathrm{rel}$",
    "part_etarel": "Particle $\\eta^\\mathrm{rel}$",
    "part_phirel": "Particle $\\phi^\\mathrm{rel}$",
    "jet_pt": "Jet $p_{\\mathrm{T}}$ [GeV]",
    "jet_eta": "Jet $\\eta$",
    "jet_phi": "Jet $\\phi$",
    "jet_mass": "Jet mass [GeV]",
}

params_to_update = {
    # --- axes ---
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    "axes.prop_cycle": cycler(
        "color",
        [
            mpl.colors.ColorConverter().to_rgba(col, DEFAULT_ALPHA)
            for col in [
                "steelblue",
                "orange",
                "forestgreen",
                "purple",
                "firebrick",
                "lightseagreen",
                "yellowgreen",
                "hotpink",
                "dimgrey",
                "olive",
            ]
        ],
    ),
    # --- figure ---
    "figure.figsize": (3.5, 2.5),
    # "figure.dpi": 130,
    # --- grid ---
    "grid.color": "black",
    "grid.alpha": 0.1,
    "grid.linestyle": "-",
    "grid.linewidth": 1,
    # --- legend ---
    "legend.fontsize": 10,
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    # --- lines ---
    "lines.linewidth": 1.5,
    "lines.markeredgewidth": 0,
    "lines.markersize": 7,
    # "lines.solid_capstyle": "round",
    # --- patches ---
    "patch.facecolor": "4C72B0",
    "patch.linewidth": 1.7,
    # --- histogram ---
    "hist.bins": 100,
    # --- font ---
    "font.family": "sans-serif",
    "font.sans-serif": "Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif",
    # --- image ---
    "image.cmap": "Greys",
}
params_to_update_dark = {
    # --- axes ---
    "axes.prop_cycle": cycler(
        "color",
        [
            mpl.colors.ColorConverter().to_rgba(col, DEFAULT_ALPHA)
            for col in [
                "dodgerblue",
                "red",
                "mediumseagreen",
                "darkorange",
                "orchid",
                "turquoise",
                "#64B5CD",
            ]
        ],
    ),
    # --- figure ---
    "figure.facecolor": "#1A1D22",
    # --- grid ---
    "grid.color": "lightgray",
    # --- lines ---
    "lines.solid_capstyle": "round",
    # --- font ---
    "font.family": "sans-serif",
    "font.sans-serif": "Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif",
    # --- image ---
    "image.cmap": "Greys",
    # --- legend ---
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    # --- xtick ---
    "xtick.direction": "in",
    "xtick.color": "white",
    # --- ytick ---
    "ytick.direction": "out",
    "ytick.color": "white",
    # --- axes.axisbelow ---
    "axes.axisbelow": True,
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "#1A1D22",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "figure.edgecolor": "#1A1D22",
    "savefig.facecolor": "#1A1D22",
    "savefig.edgecolor": "#1A1D22",
}


def reset_mpl_style():
    """Reset matplotlib rcParams to default."""
    rcParams.update(mpl.rcParamsDefault)


def set_mpl_style(darkmode=False):
    """Set matplotlib rcParams to custom configuration."""
    reset_mpl_style()
    rcParams.update(params_to_update if not darkmode else params_to_update_dark)


def save(fig, saveas, transparent=True):
    """Save a figure both as pdf and as png.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    saveas : str
        The path to save the figure to, expected to end in ".pdf".
    transparent : bool, optional
        Whether to save the figure with a transparent background, by default True
    """
    save_kwargs = dict(transparent=transparent, dpi=300, bbox_inches="tight")
    # create the directory if it does not exist
    if not Path(saveas).parent.exists():
        print(f"Creating directory {Path(saveas).parent}")
        Path(saveas).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {saveas}")
    fig.savefig(saveas, **save_kwargs)
    fig.savefig(str(saveas).replace(".pdf", ".png"), **save_kwargs)


# # default seaborn aesthetic
# # darkgrid + deep palette + notebook context

# # axes.axisbelow: True
# # axes.edgecolor: white
# # axes.facecolor: EAEAF2
# # axes.grid: True
# # axes.labelcolor: .15
# # axes.labelsize: 11
# # axes.linewidth: 0
# # axes.prop_cycle: cycler('color', ['4C72B0', '55A868', 'C44E52', '8172B2', 'CCB974', '64B5CD'])
# axes.prop_cycle: cycler('color', ['55A868', 'C44E52', '4C72B0', '8172B2', 'CCB974', '64B5CD'])
# # axes.prop_cycle: cycler('color', ['55A868', '9E4AC2','4C72B0' , 'C44E52', '8172B2', 'CCB974', '64B5CD'])
# # axes.titlesize: 12

# figure.facecolor: white


# # xtick.color: .15
# # xtick.direction: out
# # xtick.labelsize: 10
# # xtick.major.pad: 7
# # xtick.major.size: 0
# # xtick.major.width: 1
# # xtick.minor.size: 0
# # xtick.minor.width: .5

# # ytick.color: .15
# # ytick.direction: out
# # ytick.labelsize: 10
# # ytick.major.pad: 7
# # ytick.major.size: 0
# # ytick.major.width: 1
# # ytick.minor.size: 0
# # ytick.minor.width: .5

# # figure.facecolor: white
# # text.color: .15
# # axes.labelcolor: .15
# # legend.frameon: False
# # legend.numpoints: 1
# # legend.scatterpoints: 1
# # xtick.direction: in
# # ytick.direction: out
# # xtick.color: .15
# # ytick.color: .15
# # axes.axisbelow: True
# # image.cmap: Greys
# # font.family: sans-serif
# # font.sans-serif: Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif
# # grid.linestyle: -
# # lines.solid_capstyle: round

# # Seaborn dark parameters
# # axes.grid: False
# # axes.facecolor: EAEAF2
# # axes.edgecolor: white
# # axes.linewidth: 0
# # grid.color: white
# # xtick.major.size: 0
# # ytick.major.size: 0
# # xtick.minor.size: 0
# # ytick.minor.size: 0


def decorate_ax(
    ax,
    yscale=1.3,
    text=None,
    text_line_spacing=1.2,
    text_font_size=12,
    draw_legend=False,
    indent=0.7,
    top_distance=1.2,
    hepstyle=False,
    remove_first_ytick=False,
):
    """Helper function to decorate the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to decorate
    yscale : float, optional
        Factor by which the y-axis is scaled, by default 1.3
    text : str, optional
        Text to add to the plot, by default None
    text_line_spacing : float, optional
        Spacing between lines of text, by default 1.2
    text_font_size : int, optional
        Font size of the text, by default 12
    draw_legend : bool, optional
        Draw the legend with `frameon=False`, by default False
    indent : float, optional
        Horizontal indent, by default 0.7
    top_distance : float, optional
        Vertical indent, by default 1.2
    hepstyle : bool, optional
        Use the atlasify function to make the plot look like an ATLAS plot, by default False
    remove_first_ytick : bool, optional
        Remove the first y-tick, by default False.
        Can be useful to avoid overlap with the ratio plot ticks.
    """
    PT = 1 / 72  # 1 point in inches

    # reset the y-axis limits (if they were changed before, it can happen
    # that the y-axis is not scaled correctly. especially it happens that ymin
    # becomes 0 even after setting logscale, which raises an error below as we
    # divide by ymin for logscale)
    if yscale != 1:
        xmin, xmax = ax.get_xlim()
        ax.relim()
        ax.autoscale()
        ax.set_xlim(xmin, xmax)

        # This weird order is necessary to allow for later
        # saving in logscaled y-axis
        if ax.get_yscale() == "log":
            ymin, _ = ax.get_ylim()
            ax.set_yscale("linear")
            _, ymax = ax.get_ylim()
            ax.set_yscale("log")
            yscale = (ymax / ymin) ** (yscale - 0.99)
        else:
            ymin, ymax = ax.get_ylim()

        # scale the y-axis to avoid overlap with text
        ax.set_ylim(top=yscale * (ymax - ymin) + ymin)

    if text is None:
        pass
    elif isinstance(text, str):
        # translation from the left side of the axes (aka indent)
        trans_indent = ScaledTranslation(
            indent * text_line_spacing * PT * text_font_size,
            0,
            ax.figure.dpi_scale_trans,
        )
        # translation from the top of the axes
        trans_top = ScaledTranslation(
            0,
            -top_distance * text_line_spacing * PT * text_font_size,
            ax.figure.dpi_scale_trans,
        )

        # add each line of the tag text to the plot
        for line in text.split("\n"):
            # fmt: off
            ax.text(0, 1, line, transform=ax.transAxes + trans_top + trans_indent, fontsize=text_font_size)  # noqa: E501
            trans_top += ScaledTranslation(0, -text_line_spacing * text_font_size * PT, ax.figure.dpi_scale_trans)  # noqa: E501
            # fmt: on
    else:
        raise TypeError("`text` attribute of the plot has to be of type `str`.")

    if draw_legend:
        ax.legend(frameon=False)

    if remove_first_ytick:
        # remove the first y-tick label of the upper plot to avoid overlap with the ratio plot
        ax.set_yticks(ax.get_yticks()[1:])


def get_good_linestyles(names=None):
    """Returns a list of good linestyles.

    Parameters
    ----------
    names : list or str, optional
        List or string of the name(s) of the linestyle(s) you want to retrieve, e.g.
        "densely dotted" or ["solid", "dashdot", "densely dashed"], by default None

    Returns
    -------
    list
        List of good linestyles. Either the specified selection or the whole list in
        the predefined order.

    Raises
    ------
    ValueError
        If `names` is not a str or list.
    """
    linestyle_tuples = {
        "solid": "solid",
        "densely dashed": (0, (5, 1)),
        "densely dotted": (0, (1, 1)),
        "densely dashdotted": (0, (3, 1, 1, 1)),
        "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
        "dotted": (0, (1, 1)),
        "dashed": (0, (5, 5)),
        "dashdot": "dashdot",
        "loosely dashed": (0, (5, 10)),
        "loosely dotted": (0, (1, 10)),
        "loosely dashdotted": (0, (3, 10, 1, 10)),
        "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
        "dashdotted": (0, (3, 5, 1, 5)),
        "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    }

    default_order = [
        "solid",
        "densely dotted",
        "densely dashed",
        "densely dashdotted",
        "densely dashdotdotted",
        "dotted",
        "dashed",
        "dashdot",
        # "loosely dotted",
        # "loosely dashed",
        # "loosely dashdotted",
        # "loosely dashdotdotted",
        "dashdotted",
        "dashdotdotted",
    ]
    if names is None:
        names = default_order * 3
    elif isinstance(names, str):
        return linestyle_tuples[names]
    elif not isinstance(names, list):
        raise ValueError("Invalid type of `names`, has to be a list of strings or a string.")
    return [linestyle_tuples[name] for name in names]


def get_col(i):
    """Get the i-th color from the default color cycle."""
    return rcParams["axes.prop_cycle"].by_key()["color"][i % len(rcParams["axes.prop_cycle"])]


def get_label(name):
    """Get the label for the given name."""
    return DEFAULT_LABELS[name]


def get_ax_and_fig_2x5(ratio=True, figsize=(18, 6)):
    """Create a 2x5 grid of axes for plotting histograms and ratios. The axes are returned as a
    2x10 grid, where the first row contains the histogram plots and the second row contains the
    ratio plots.

    Parameters:
    ----------
    ratio : bool, optional
        Whether to include a ratio plot. Default is True.
    figsize : tuple, optional
        Size of the figure.
    """

    if ratio:
        gridspec = dict(hspace=0.0, height_ratios=[1, 0.3, 0.3, 1, 0.3])
        fig, ax = plt.subplots(5, 5, figsize=figsize, gridspec_kw=gridspec)
        # make third row invisible
        for i in range(5):
            ax[2, i].axis("off")
        # remove the dummy axes in the middle
        axes = np.concatenate([ax[:2, :], ax[3:, :]], axis=1)
    else:
        fig, ax = plt.subplots(2, 5, figsize=figsize)
        axes = ax.flatten()
    return fig, axes


def get_ax_and_fig_for_ratio_plot(figsize=(4, 3)):
    """Returns fig, axes for a ratio plot. ax[0] is the histogram, ax[1] is the ratio.

    Parameters:
    ----------
    figsize : tuple, optional
        Size of the figure.
    """

    gridspec = dict(hspace=0.0, height_ratios=[1, 0.3])
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw=gridspec)
    # make third second ax invisible
    # ax[1].axis("off")
    # remove the dummy axes in the middle
    # axes = np.concatenate([ax[0], ax[-1]], axis=1)
    return fig, ax


def save_two_subplots(
    fig,
    ax1,
    ax2,
    saveas: str,
    expanded_x=1.05,
    expanded_y=1.05,
):
    """Save two subplots of a figure to a single file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the subplots.
    ax1 : matplotlib.axes.Axes
        The first subplot.
    ax2 : matplotlib.axes.Axes
        The second subplot.
    saveas : str
        The path to save the figure to.
    expanded_x : float, optional
        Factor by which to expand the bounding box in the x-direction, by default 1.05
    expanded_y : float, optional
        Factor by which to expand the bounding box in the y-direction, by default 1.05
    """
    bbox1 = ax1.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    bbox2 = ax2.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    bbox_total = Bbox.union([bbox1, bbox2])
    fig.savefig(saveas, bbox_inches=bbox_total.expanded(expanded_x, expanded_y))


def save_subplot(
    fig,
    ax,
    saveas: str,
    expanded_x=1.05,
    expanded_y=1.05,
):
    """Save a single subplot of a figure to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the subplot.
    ax : matplotlib.axes.Axes
        The subplot to save.
    saveas : str
        The path to save the figure to.
    expanded_x : float, optional
        Factor by which to expand the bounding box in the x-direction, by default 1.05
    expanded_y : float, optional
        Factor by which to expand the bounding box in the y-direction, by default 1.05
    """
    bbox = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(saveas, bbox_inches=bbox.expanded(expanded_x, expanded_y))
