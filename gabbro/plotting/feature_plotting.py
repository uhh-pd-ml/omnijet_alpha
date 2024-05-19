import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vector
from matplotlib.lines import Line2D

import gabbro.plotting.utils as plot_utils

vector.register_awkward()


def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)


def get_bin_centers_and_bin_heights_from_hist(hist):
    """Return the bin centers and bin heights from a histogram.

    Parameters
    ----------
    hist : tuple
        The output of matplotlib hist.

    Returns
    -------
    bin_centers : array-like
        The bin centers.
    bin_heights : array-like
        The bin heights.
    """
    bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
    bin_heights = hist[0]
    return bin_centers, bin_heights


def plot_hist_with_ratios(
    comp_dict: dict,
    bins: np.ndarray,
    ax_upper: plt.Axes,
    ax_ratio: plt.Axes = None,
    ref_dict: dict = None,
    ratio_range: tuple = None,
    xlabel: str = None,
    logy: bool = False,
    leg_loc: str = "best",
    underoverflow: bool = True,
    leg_title: str = None,
    leg_ncols: int = 1,
    return_hist_curve: bool = False,
):
    """Plot histograms of the reference and comparison arrays, and their ratio.

    Parameters:
    ----------
    ax_upper : plt.Axes
        Axes for the upper panel.
    ax_ratio : plt.Axes
        Axes for the ratio panel.
    ref_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the reference array.
    comp_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the comparison arrays.
    bins : np.ndarray
        Bin edges for the histograms.
    ratio_range : tuple, optional
        Range of the y-axis for the ratio plot.
    xlabel : str, optional
        Label for the x-axis.
    logy : bool, optional
        Whether to plot the y-axis in log scale.
    leg_loc : str, optional
        Location of the legend.
    underoverflow : bool, optional
        Whether to include underflow and overflow bins. Default is True.
    leg_title : str, optional
        Title of the legend.
    leg_ncols : int, optional
        Number of columns in the legend. Default is 1.
    return_hist_curve : bool, optional
        Whether to return the histogram curves in a dict. Default is False.

    Returns
    -------
    hist_curve_dict : dict
        Dict with {id: (bin_centers, bin_heights), ...} of the histogram curves.
        Only returned if `return_hist_curve` is True. Both bin_centers and bin_heights
        are array-like.
    """

    legend_handles = []
    hist_curve_dict = {}

    if ref_dict is not None:
        ref_arr = list(ref_dict.values())[0]
        ref_label = list(ref_dict.keys())[0]
        kwargs_ref = dict(histtype="stepfilled", color="k", alpha=0.25, label=ref_label)

    if leg_title is not None:
        # plot empty array with alpha 0 to create a legend entry
        ax_upper.hist([], alpha=0, label=leg_title)

    kwargs_common = dict(bins=bins, density=True)
    if ref_dict is not None:
        hist_ref = ax_upper.hist(binclip(ref_arr["arr"], bins), **kwargs_common, **kwargs_ref)

    if ax_ratio is not None:
        ax_ratio.axhline(1, color="black", linestyle="--", lw=1)

    # loop over entries in comp_dict and plot them
    for i, (arr_id, arr_dict) in enumerate(comp_dict.items()):
        kwargs_comp = dict(histtype="step") | arr_dict.get("hist_kwargs", {})
        if "linestyle" in kwargs_comp:
            if kwargs_comp["linestyle"] == "dotted":
                kwargs_comp["linestyle"] = plot_utils.get_good_linestyles("densely dotted")
        hist_comp = ax_upper.hist(binclip(arr_dict["arr"], bins), **kwargs_common, **kwargs_comp)
        if return_hist_curve:
            hist_curve_dict[arr_id] = get_bin_centers_and_bin_heights_from_hist(hist_comp)
        legend_handles.append(
            Line2D(
                [],
                [],
                color=kwargs_comp.get("color", "C1"),
                lw=kwargs_comp.get("lw", 1),
                label=kwargs_comp.get("label", arr_id),
                linestyle=kwargs_comp.get("linestyle", "-"),
            )
        )
        if ax_ratio is not None:
            # calculate and plot ratio
            ratio = hist_comp[0] / hist_ref[0]
            # duplicate the first entry to avoid a gap in the plot (due to step plot)
            ratio = np.append(np.array(ratio[0]), np.array(ratio))
            bin_edges = hist_ref[1]
            ax_ratio.step(bin_edges, ratio, where="pre", **arr_dict.get("hist_kwargs", {}))

    ax_upper.legend(
        # handles=legend_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    # re-do legend, with the first handle kep and the others replaced by the new list
    old_handles, old_labels = ax_upper.get_legend_handles_labels()
    new_handles = old_handles[:1] + legend_handles if ref_dict is not None else legend_handles
    ax_upper.legend(
        handles=new_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    ax_upper.set_ylabel("Normalized")

    ax_upper.set_xlim(bins[0], bins[-1])

    if ax_ratio is not None:
        ax_ratio.set_xlim(bins[0], bins[-1])
        ax_upper.set_xticks([])

    if ratio_range is not None:
        ax_ratio.set_ylim(*ratio_range)
    if xlabel is not None:
        if ax_ratio is not None:
            ax_ratio.set_xlabel(xlabel)
        else:
            ax_upper.set_xlabel(xlabel)
    if logy:
        ax_upper.set_yscale("log")
    return hist_curve_dict if return_hist_curve else None


def plot_two_jet_versions(const1, const2, label1="version1", label2="version2", title=None):
    """Plot the constituent and jet features for two jet collections.

    Parameters:
    ----------
    const1 : awkward array
        Constituents of the first jet collection.
    const2 : awkward array
        Constituents of the second jet collection.
    title : str, optional
        Title of the plot.
    """

    jets1 = ak.sum(const1, axis=1)
    jets2 = ak.sum(const2, axis=1)

    fig, axarr = plt.subplots(4, 4, figsize=(12, 8))
    histkwargs = dict(bins=100, density=True, histtype="step")

    part_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(part_feats):
        axarr[0, i].hist(ak.flatten(const1[feat]), **histkwargs, label=label1)
        axarr[0, i].hist(ak.flatten(const2[feat]), **histkwargs, label=label1)
        axarr[0, i].set_xlabel(f"Constituent {feat}")
        # plot the difference
        axarr[1, i].hist(
            ak.flatten(const2[feat]) - ak.flatten(const1[feat]),
            **histkwargs,
            label=f"{label2} - {label1}",
        )
        axarr[1, i].set_xlabel(f"Constituent {feat} resolution")

    jet_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(jet_feats):
        axarr[2, i].hist(getattr(jets1, feat), **histkwargs, label=label1)
        axarr[2, i].hist(getattr(jets2, feat), **histkwargs, label=label2)
        axarr[2, i].set_xlabel(f"Jet {feat}")
        axarr[3, i].hist(
            getattr(jets2, feat) - getattr(jets1, feat), **histkwargs, label=f"{label2} - {label1}"
        )
        axarr[3, i].set_xlabel(f"Jet {feat} resolution")

    axarr[0, 0].legend(frameon=False)
    axarr[1, 0].legend(frameon=False)
    axarr[2, 0].legend(frameon=False)
    axarr[3, 0].legend(frameon=False)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    # plt.show()
    return fig, axarr


def plot_features(
    ak_array_dict,
    names=None,
    label_prefix=None,
    flatten=True,
    histkwargs=None,
    legend_only_on=None,
    legend_kwargs={},
    ax_rows=1,
    decorate_ax_kwargs={},
    bins_dict=None,
    logscale_features=None,
    colors=None,
    ax_size=(3, 2),
):
    """Plot the features of the constituents or jets.

    Parameters:
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or jets to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    logscale_features : list, optional
        List of features to plot in log scale, of "all" to plot all features in log scale.
    colors : list, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
    ax_size : tuple, optional
        Size of the axes. Default is (3, 2).
    """

    default_hist_kwargs = {"density": True, "histtype": "step", "bins": 100}

    # setup colors
    if colors is not None:
        if len(colors) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors = colors + [f"C{i}" for i in range(len(ak_array_dict) - len(colors))]

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        histkwargs = default_hist_kwargs | histkwargs

    # create the bins dict
    if bins_dict is None:
        bins_dict = {}
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names:
        if name not in bins_dict:
            bins_dict[name] = histkwargs["bins"]

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    ax_cols = len(names) // ax_rows + 1 if len(names) % ax_rows > 0 else len(names) // ax_rows

    fig, axarr = plt.subplots(
        ax_rows, ax_cols, figsize=(ax_size[0] * ax_cols, ax_size[1] * ax_rows)
    )
    if len(names) == 1:
        axarr = [axarr]
    else:
        axarr = axarr.flatten()

    legend_handles = []
    legend_labels = []

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        color = colors[i_label] if colors is not None else f"C{i_label}"
        legend_labels.append(label)
        for i, (feat, feat_label) in enumerate(names.items()):
            if flatten:
                values = ak.flatten(getattr(ak_array, feat))
            else:
                values = getattr(ak_array, feat)

            if not isinstance(bins_dict[feat], int):
                values = binclip(values, bins_dict[feat])

            _, _, patches = axarr[i].hist(values, **histkwargs, bins=bins_dict[feat], color=color)
            axarr[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=label,
                        linestyle=patches[0].get_linestyle(),
                    )
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False
    for i, (_ax, feat_name) in enumerate(zip(axarr, names.keys())):
        if legend_only_on is None:
            _ax.legend(**legend_kwargs)
        else:
            if i == legend_only_on:
                _ax.legend(**legend_kwargs)

        if (logscale_features is not None and feat_name in logscale_features) or (
            logscale_features == "all"
        ):
            _ax.set_yscale("log")
        plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

    fig.tight_layout()
    return fig, axarr


def plot_features_pairplot(
    arr,
    names=None,
    pairplot_kwargs={},
    input_type="ak_constituents",
):
    """Plot the features of the constituents or jets using a pairplot.

    Parameters:
    ----------
    arr : awkward array or numpy array
        Constituents or jets.
    part_names : list or dict, optional
        List of names of the features to plot, or dict of {"name": "label", ...}.
    pairplot_kwargs : dict, optional
        Keyword arguments passed to sns.pairplot.
    input_type : str, optional
        Type of the input array. Can be "ak_constituents", "ak_jets", or "np_flat".
        "ak_constituents" is an awkward array of jet constituents of shape `(n_jets, <var>, n_features)`.
        "ak_jets" is an awkward array of jets of shape `(n_jets, n_features)`.
        "np_flat" is a numpy array of shape `(n_entries, n_features)`


    Returns:
    --------
    pairplot : seaborn.axisgrid.PairGrid
        Pairplot object of the features.
    """

    if isinstance(names, list):
        names = {name: name for name in names}

    sns.set_style("dark")
    # create a dataframe from the awkward array
    if input_type == "ak_constituents":
        df = pd.DataFrame(
            {feat_label: ak.flatten(getattr(arr, feat)) for feat, feat_label in names.items()}
        )
    elif input_type == "ak_jets":
        df = pd.DataFrame({feat_label: getattr(arr, feat) for feat, feat_label in names.items()})
    elif input_type == "np_flat":
        df = pd.DataFrame(
            {feat_label: arr[:, i] for i, (feat, feat_label) in enumerate(names.items())}
        )
    else:
        raise ValueError(f"Invalid input_type: {input_type}")
    pairplot = sns.pairplot(df, kind="hist", **pairplot_kwargs)
    plt.show()

    # reset the style
    plt.rcdefaults()

    return pairplot
