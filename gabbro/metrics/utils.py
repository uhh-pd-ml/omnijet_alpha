"""Utility functions for metrics."""

import numpy as np
import scipy


def quantiled_kl_divergence(sample_ref: np.ndarray, sample_approx: np.ndarray, n_bins: int = 30):
    """Calculate the KL divergence using quantiles on sample_ref to define the bounds.

    Parameters
    ----------
    sample_ref : np.ndarray
        The first sample to compare (this is the reference, so in the context of
        jet generation, those are the real jets).
    sample_approx : np.ndarray
        The second sample to compare (this is the model/approximation, so in the
        context of jet generation, those are the generated jets).
    n_bins : int
        The number of bins to use for the histogram. Those bins are defined by
        equiprobably quantiles of sample_ref.
    """
    bins = np.quantile(sample_ref, np.linspace(0.001, 0.999, n_bins))
    hist_1 = np.histogram(sample_ref, bins, density=True)[0] + 1e-8
    hist_2 = np.histogram(sample_approx, bins, density=True)[0] + 1e-8
    return scipy.stats.entropy(hist_1, hist_2)


def calc_quantiled_kl_divergence_for_dict(
    dict_reference: dict,
    dict_approx: dict,
    names: list,
    n_bins: int = 30,
):
    """Calculate the quantiled KL divergence for two dictionaries of samples.

    Parameters
    ----------
    dict_reference : dict
        The first dictionary of samples.
    dict_approx : dict
        The second dictionary of samples.
    names : list
        The names of the samples to compare. All names must be included in both dicts.
    """
    # loop over the names and calculate the quantiled kld for each name
    klds = {}
    for name in names:
        klds[name] = quantiled_kl_divergence(
            sample_ref=np.array(dict_reference[name]),
            sample_approx=np.array(dict_approx[name]),
            n_bins=n_bins,
        )
    return klds


def calc_accuracy(preds, labels, verbose=False):
    """Calculates accuracy and AUC.

    Parameters
    ----------
    preds : array-like
        Classifier scores. Tensor of shape (n_samples, n_classes).
    labels : array-like
        Array with the true labels (one-hot encoded). Tensor of shape (n_samples, n_classes).

    Returns
    -------
    accuracy : float
        Accuracy.
    """
    accuracy = (np.argmax(preds, axis=1) == np.argmax(labels, axis=1)).mean()

    return accuracy


def calc_rejection(scores, labels, verbose=False, sig_eff=0.3):
    """Calculates the R30 metric.

    Parameters
    ----------
    scores : array-like
        Classifier scores (probability of being signal). Array of shape (n_samples,).
    labels : array-like
        Array with the true labels (0 or 1). Array of shape (n_samples,).
    sig_eff : float, optional
        Signal efficiency at which to calculate the rejection.

    Returns
    -------
    rejection : float
        Rejection metric value.
    cut_value : float
        Cut value for this rejection.
    """
    is_signal = labels == 1
    cut_value = np.percentile(scores[is_signal], 100 - sig_eff * 100)
    background_efficiency = np.sum(scores[~is_signal] > cut_value) / np.sum(~is_signal)
    if verbose:
        print(f"cut_value = {cut_value}")
        print(f"background_efficiency = {background_efficiency}")
    rejection = 1 / background_efficiency
    return rejection, cut_value
