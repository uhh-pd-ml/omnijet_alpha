import awkward as ak
import numpy as np
import torch
import vector

from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

vector.register_awkward()


def p4s_from_ptetaphimass(
    ak_arr,
    field_name_pt="part_pt",
    field_name_eta="part_etarel",
    field_name_phi="part_phirel",
    field_name_mass="part_mass",
):
    """Create a Momentum4D array from pt, eta, phi, mass fields in an awkward array.

    Parameters
    ----------
    ak_arr : ak.Array
        Array with fields part_pt, part_etarel, part_phirel, part_mass.
    field_name_pt : str, optional
        Name of the field containing the transverse momentum, by default "part_pt".
    field_name_eta : str, optional
        Name of the field containing the pseudorapidity, by default "part_etarel".
    field_name_phi : str, optional
        Name of the field containing the azimuthal angle, by default "part_phirel".
    field_name_mass : str, optional
        Name of the field containing the mass, by default
    """
    return ak.zip(
        {
            "pt": ak_arr[field_name_pt],
            "eta": ak_arr[field_name_eta],
            "phi": ak_arr[field_name_phi],
            "mass": ak_arr[field_name_mass]
            if field_name_mass in ak_arr.fields
            else ak.zeros_like(ak_arr[field_name_pt]),
        },
        with_name="Momentum4D",
    )


def ak_pad(x: ak.Array, maxlen: int, axis: int = 1, fill_value=0, return_mask=False):
    """Function to pad an awkward array to a specified length. The array is padded along the
    specified axis.

    Parameters
    ----------
    x : awkward array
        Array to pad.
    maxlen : int
        Length to pad to.
    axis : int, optional
        Axis along which to pad. Default is 1.
    fill_value : float or int, optional
        Value to use for padding. Default is 0.
    return_mask : bool, optional
        If True, also return a mask array indicating which values are padded.
        Default is False.
        If the input array has fields, the mask is created from the first field.

    Returns
    -------
    awkward array
        Padded array.
    mask : awkward array
        Mask array indicating which values are padded. Only returned if return_mask is True.
    """
    padded_x = ak.fill_none(ak.pad_none(x, maxlen, axis=axis, clip=True), fill_value)
    if return_mask:
        if len(x.fields) >= 1:
            mask = ak.ones_like(x[x.fields[0]], dtype="bool")
        else:
            mask = ak.ones_like(x, dtype="bool")
        mask = ak.fill_none(ak.pad_none(mask, maxlen, axis=axis, clip=True), False)
        return padded_x, mask
    return padded_x


def combine_ak_arrays(*arrays):
    """Function to combine multiple awkward arrays. The arrays should have different fields.

    Parameters
    ----------
    *arrays : ak.Array
        Input arrays to combine.

    Returns
    -------
    ak.Array
        Combined array.
    """
    combined_fields = {}
    for arr in arrays:
        if arr is None:
            continue
        if set(combined_fields.keys()) & set(arr.fields):
            dict_with_fields = {f"arr{i}": arr.fields for i, arr in enumerate(arrays)}
            raise ValueError(
                "You are trying to merge multiple ak.Arrays but they have common field names. "
                f"The common field names are: {set(combined_fields.keys()) & set(arr.fields)} "
                f"The individual field names are: {dict_with_fields}"
            )
        combined_fields.update({field: arr[field] for field in arr.fields})
    return ak.Array(combined_fields)


def np_to_ak(x: np.ndarray, names: list, mask: np.ndarray = None, dtype="float32"):
    """Function to convert a numpy array and its mask to an awkward array. The features
    corresponding to the names are assumed to correspond to the last axis of the array.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    names : list
        List of field names (corresponding to the features in x along the last dimension).
    mask : np.ndarray, optional
        Mask array. Default is None. If x is an array of shape (n, m, k), the mask should
        be of shape (n, m).
    dtype : str, optional
        Data type to convert the values to. Default is "float32".
    """

    if mask is None:
        mask = np.ones_like(x[..., 0], dtype="bool")

    return ak.Array(
        {
            name: ak.values_astype(
                ak.drop_none(ak.mask(ak.Array(x[..., i]), mask != 0)),
                dtype,
            )
            for i, name in enumerate(names)
        }
    )


def ak_to_np_stack(ak_array: ak.Array, names: list = None, axis: int = -1):
    """Function to convert an awkward array to a numpy array by stacking the values of the
    specified fields. This is much faster than ak.to_numpy(ak_array) for large arrays.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    names : list, optional
        List of field names to convert. Default is None.
    axis : int, optional
        Axis along which to stack the values. Default is -1.
    """
    if names is None:
        raise ValueError("names must be specified")
    return ak.to_numpy(
        np.stack(
            [ak.to_numpy(ak.values_astype(ak_array[name], "float32")) for name in names],
            axis=axis,
        )
    )


def np_PtEtaPhi_to_Momentum4D(arr, mask, log_pt=False):
    """Convert numpy array with 4-momenta to ak array of Momentum4D objects.
    NOTE: the input array is assumed to be in (pT, eta, phi) format, thus mass = 0.

    Expects an array of shape (batch_size, num_particles, 3)
    where the last dimension is (pt, eta, phi)

    Returns an ak array of shape (batch_size, var, 4) of Momentum4D objects

    If log_pt is True, the corresponding variable is exponentiated
    before being passed to Momentum4D

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (batch_size, num_particles, 3)
    mask : np.ndarray
        Mask array of shape (batch_size, num_particles)
    log_pt : bool, optional
        Whether to exponentiate pt, by default False

    Returns
    -------
    ak.Array
        Array of Momentum4D objects
    """

    p4 = ak.zip(
        {
            "pt": np.clip(arr[:, :, 0], 0, None) if not log_pt else np.exp(arr[:, :, 0]),
            "eta": arr[:, :, 1],
            "phi": arr[:, :, 2],
            "mass": ak.zeros_like(arr[:, :, 0]),
        },
        with_name="Momentum4D",
    )
    # mask the array
    ak_mask = ak.Array(mask)
    return ak.drop_none(ak.mask(p4, ak_mask == 1))


def ak_select_and_preprocess(ak_array: ak.Array, pp_dict=None, inverse=False):
    """Function to select and pre-process fields from an awkward array.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    pp_dict : dict, optional
        Dictionary with pre-processing values for each field. Default is None.
        The dictionary should have the following format:
        {
            "field_name_1": {"multiply_by": 1, "subtract_by": 0, "func": "np.log"},
            "field_name_2": {"multiply_by": 1, "subtract_by": 0, "func": None},
            ...
        }
    inverse : bool, optional
        If True, the inverse of the pre-processing is applied. Default is False.
    """
    if pp_dict is None:
        pp_dict = {}

    # define initial mask as all True
    first_feat = list(pp_dict.keys())[0]
    selection_mask = ak.ones_like(ak_array[first_feat], dtype="bool")

    for name, params in pp_dict.items():
        if params is None:
            pp_dict[name] = {
                "subtract_by": 0,
                "multiply_by": 1,
                "func": None,
                "inv_func": None,
                "larger_than": None,
                "smaller_than": None,
                "binning": None,
                "bin_edges": None,
                "clip_min": None,
                "clip_max": None,
            }
        else:
            if "subtract_by" not in params:
                pp_dict[name]["subtract_by"] = 0
            if "multiply_by" not in params:
                pp_dict[name]["multiply_by"] = 1
            if "func" not in params:
                pp_dict[name]["func"] = None
            if "inv_func" not in params:
                pp_dict[name]["inv_func"] = None
            if "larger_than" not in params:
                pp_dict[name]["larger_than"] = None
            if "smaller_than" not in params:
                pp_dict[name]["smaller_than"] = None
            if "bin_edges" not in params:
                pp_dict[name]["bin_edges"] = None
            if "binning" not in params:
                pp_dict[name]["binning"] = None
            elif pp_dict[name]["binning"] is not None:
                # convert tuple of (start, end, n_bins) to np.linspace
                start, stop, n_bins = pp_dict[name]["binning"]
                if pp_dict[name].get("bin_edges") is None:
                    pp_dict[name]["bin_edges"] = np.linspace(start, stop, int(n_bins))
                print(
                    f"Applying binning to field {name} with np.linspace({start}, {stop}, {n_bins})"
                )
            if "clip_min" not in params:
                pp_dict[name]["clip_min"] = None
            if "clip_max" not in params:
                pp_dict[name]["clip_max"] = None

            if pp_dict[name]["clip_min"] is not None or pp_dict[name]["clip_max"] is not None:
                logger.warning(
                    f"You are clipping the values of the feature '{name}' with "
                    f"clip_min = {pp_dict[name]['clip_min']} and "
                    f"clip_max = {pp_dict[name]['clip_max']}"
                    ". Make sure this is intended. THIS IS NOT INVERTIBLE."
                )

            if pp_dict[name]["clip_min"] is not None and pp_dict[name]["clip_max"] is not None:
                if pp_dict[name]["clip_min"] > pp_dict[name]["clip_max"]:
                    raise ValueError(
                        "clip_min must be smaller than clip_max."
                        f"You have clip_min = {pp_dict[name]['clip_min']} and "
                        f"clip_max = {pp_dict[name]['clip_max']} for field {name}."
                    )

            if pp_dict[name]["func"] is not None:
                if pp_dict[name]["inv_func"] is None:
                    raise ValueError(
                        "If a function is specified, an inverse function must also be specified."
                    )
            else:
                if pp_dict[name]["inv_func"] is not None:
                    raise ValueError(
                        "If an inverse function is specified, a function must also be specified."
                    )
        # apply selection cuts
        if pp_dict[name].get("larger_than") is not None:
            selection_mask = selection_mask & (ak_array[name] > pp_dict[name]["larger_than"])
        if pp_dict[name].get("smaller_than") is not None:
            selection_mask = selection_mask & (ak_array[name] < pp_dict[name]["smaller_than"])

    if inverse:
        return ak.Array(
            {
                name: (
                    eval(params["inv_func"])(  # nosec
                        getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                    )
                    if params["inv_func"]
                    else getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                )
                for name, params in pp_dict.items()
            }
        )
    return ak.Array(
        {
            name: (
                ak_clip(
                    (
                        apply_binning(
                            eval(params["func"])(  # nosec
                                getattr(ak_array, name)
                            ),
                            params["bin_edges"],
                        )[selection_mask]
                        if params["func"]
                        else apply_binning(getattr(ak_array, name), params["bin_edges"])[
                            selection_mask
                        ]
                    ),
                    params["clip_min"],
                    params["clip_max"],
                )
                - params["subtract_by"]
            )
            * params["multiply_by"]
            for name, params in pp_dict.items()
        }
    )


# define a function to sort ak.Array by pt
def sort_by_pt(constituents: ak.Array, ascending: bool = False):
    """Sort ak.Array of jet constituents by the pt
    Args:
        constituents (ak.Array): constituents array that should be sorted by pt.
            It should have a pt attribute.
        ascending (bool, optional): If True, the first value in each sorted
            group will be smallest; if False, the order is from largest to
            smallest. Defaults to False.
    Returns:
        ak.Array: sorted constituents array
    """
    if isinstance(constituents, ak.Array):
        try:
            temppt = constituents.pt
        except AttributeError:
            raise AttributeError(
                "Trying to sort an ak.Array without a pt attribute. Please check the input."
            )
    indices = ak.argsort(temppt, axis=1, ascending=ascending)
    return constituents[indices]


def ak_smear(arr, sigma=0, seed=42):
    """Helper function to smear an array of values by a given sigma.

    Parameters
    ----------
    arr : awkward array
        The array to smear
    sigma : float, optional
        The sigma of the smearing, by default 0 (i.e. no smearing)
    seed : int, optional
        Seed for the random number generator, by default 42
    """
    # Convert it to a 1D numpy array and perform smearing
    numpy_arr = ak.to_numpy(arr.layout.content)

    if sigma != 0:
        rng = np.random.default_rng(seed)
        numpy_arr = rng.normal(numpy_arr, sigma)

    # Convert it back to awkward form
    return ak.Array(ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout))


def ak_clip(arr, clip_min=None, clip_max=None):
    """Helper function to clip the values of an array.

    Parameters
    ----------
    arr : awkward array
        The array to clip
    clip_min : float, optional
        Minimum value to clip to, by default None
    clip_max : float, optional
        Maximum value to clip to, by default None
    """
    ndim = arr.ndim
    # Convert it to a 1D numpy array and perform clipping
    if ndim > 1:
        numpy_arr = ak.to_numpy(arr.layout.content)
    else:
        numpy_arr = ak.to_numpy(arr)

    if clip_min is not None:
        numpy_arr = np.clip(numpy_arr, clip_min, None)

    if clip_max is not None:
        numpy_arr = np.clip(numpy_arr, None, clip_max)

    if ndim > 1:
        # Convert it back to awkward form
        return ak.Array(
            ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout)
        )
    return numpy_arr


def ak_subtract(arr1, arr2):
    """Helper function to subtract two awkward arrays with names fields, i.e. arr1 - arr2

    Parameters
    ----------
    arr1 : ak.Array
        First array
    arr2 : ak.Array
        Second array

    Returns
    -------
    ak.Array
        Array with the fields of arr1 - arr2
    """

    if arr1.fields != arr2.fields:
        raise ValueError(
            "The two arrays do not have the same fields. Array 1 has fields: "
            f"{arr1.fields}, while array 2 has fields: {arr2.fields}"
        )

    if len(arr1) != len(arr2):
        raise ValueError(
            "The two arrays do not have the same length. Array 1 has length: "
            f"{len(arr1)}, while array 2 has length: {len(arr2)}"
        )

    if len(arr1.fields) == 0 or len(arr2.fields) == 0:
        raise ValueError("One or both arrays have no fields.")

    return ak.Array({name: getattr(arr1, name) - getattr(arr2, name) for name in arr1.fields})


def ak_mean(arr, axis=None):
    """Helper function to calculate the mean of an awkward array with field names along a certain
    axis.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the mean of
    axis : int, optional
        Axis along which to calculate the mean, by default None, which
        calculates the mean over all dimensions.

    Returns
    -------
    dict
        Dictionary with the mean of each field in the array. If the mean is still
        an awkward array, the values of the dict will be awkward arrays as well.
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError("Input axis must be an integer.")
        # elif axis < 0 or axis >= arr.ndim:
        #     raise ValueError("Input axis is out of range.")

    return {name: ak.mean(getattr(arr, name), axis=axis) for name in arr.fields}


def ak_abs(arr):
    """Helper function to calculate the absolute value of an awkward array with field names.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the absolute value of

    Returns
    -------
    ak.Array
        Array with the absolute values of each field
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")
    if len(arr.fields) == 0:
        raise ValueError("Input arr has no fields.")

    return ak.Array({name: np.abs(getattr(arr, name)) for name in arr.fields})


def count_appearances(arr, mask, count_up_to: int = 10):
    """
    Parameters
    ----------
    arr : np.ndarray
        Array of integers, shape (n_jets, n_constituents)
    mask : np.ndarray
        Mask array, shape (n_jets, n_constituents)
    count_up_to : int, optional
        The maximum number of appearances to check for, by default 10

    Returns
    -------
    np.ndarray
        Array of shape (n_jets, n_tokens) containing the counts of each token.
        I.e. if the maximum token number is 5, the array will have 5 columns
        indicating how many times each token appears in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the number of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the fraction of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    """
    # fill the masked values with one above the maximum value in the array
    arr = np.where(mask != 0, arr, np.max(arr) + 1)

    # Count the occurrences of each integer in each row
    counts = np.array([np.bincount(row) for row in arr])
    # remove the last column, which is the count of the maximum (fill) value
    counts = counts[:, :-1]

    # calculate how many tokens appear 0, 1, 2, 3, ... times
    n_token_appearances = []
    for i in range(count_up_to + 1):
        n_token_appearances.append(np.sum(np.array(counts) == i, axis=1))

    # calculate the percentages of tokens that appear 0, 1, 2, 3, ... times
    n_tokens_total = np.sum(mask, axis=1)
    frac_token_appearances = np.array(
        [n * i / n_tokens_total for i, n in enumerate(n_token_appearances)]
    )

    return counts, np.array(n_token_appearances).T, frac_token_appearances.T


def calc_additional_kinematic_features(ak_particles):
    """Takes in an awkward array of particles and calculates additional kinematic features. The
    initial ak array should contain part_pt, part_etarel, part_phirel.

    Parameters
    ----------
    ak_particles : ak.Array
        Array of particles with part_pt, part_etarel, part_phirel fields

    Returns
    -------
    ak.Array
        Array with additional kinematic features
    """

    # check if all features are included
    required_fields = ["part_pt", "part_etarel", "part_phirel"]
    if not all([feat in ak_particles.fields for feat in required_fields]):
        raise ValueError(
            "Not all required features are present in the input array."
            f"Required features are: {required_fields}"
        )
    p4s = ak.zip(
        {
            "pt": ak_particles.part_pt,
            "eta": ak_particles.part_etarel,
            "phi": ak_particles.part_phirel,
            "mass": ak_particles.part_mass
            if "part_mass" in ak_particles.fields
            else ak.zeros_like(ak_particles.part_pt),
        },
        with_name="Momentum4D",
    )
    p4s_jet = ak.sum(p4s, axis=1)
    return ak.Array(
        {
            "part_px": p4s.px,
            "part_py": p4s.py,
            "part_pz": p4s.pz,
            "part_ptrel": ak_particles.part_pt / p4s_jet.pt,
            "part_energy": p4s.energy,
            "part_energy_raw": p4s.energy,  # this is a workaround to be able to use the energy twice (once for lorentz vectors, once as particle feature)
            "part_erel": p4s.energy / p4s_jet.energy,
            "part_deltaR": p4s.deltaR(p4s_jet),
        }
    )


def apply_binning(arr, bin_edges, return_bin_centers=False):
    """Helper function to apply a certain binning to an array. Values outside the bin edges are
    clipped to the bin edges.

    Parameters
    ----------
    arr : np.ndarray or ak.Array
        Array to bin. If ak.Array, the array is flattened and then unflattened again.
    bin_edges : np.ndarray
        Array of bin edges.
    return_bin_centers : bool, optional
        If True, also return the bin centers. Default is False.

    Returns
    -------
    np.ndarray
        Binned array. If bin_edges is None, the input array is returned.
    np.ndarray
        Bin centers. Only returned if return_bin_centers is True.
    """
    if bin_edges is None:
        return arr
    # flatten the array to use numpy functions
    # check if it is an awkward array with nested structure
    counts = None
    if isinstance(arr, ak.Array):
        if arr.ndim > 1:
            counts = ak.num(arr)
            arr = ak.flatten(arr)
    # clip the values to the bin edges
    arr = np.clip(arr, bin_edges[0], bin_edges[-1])
    bin_indices = np.digitize(arr, bin_edges, right=False)
    # make max index one smaller than the number of bins
    bin_indices = np.where(bin_indices == len(bin_edges), len(bin_edges) - 1, bin_indices)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Replace each bin index with the corresponding bin center
    binned_arr = bin_centers[bin_indices - 1]
    if counts is not None:
        binned_arr = ak.unflatten(binned_arr, counts)
    if return_bin_centers:
        return binned_arr, bin_centers
    return binned_arr


def convert_torch_token_sequence_with_stop_token_to_ak(tensor, stop_value):
    """Convert a torch tensor with sequences to an awkward array of variable length sequences using
    the stop_value. The stop_value indicates the end of the sequence.

    E.g. if the input tensor is:
    ```
    [
        [0, 2, 1, 7, 9],
        [0, 4, 7, 9, 0],
    ]
    ```
    Then the output will be (assuming `stop_value=7`):
    ```
    [
        [0, 2, 1],
        [0, 4],
    ]
    ```

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with sequences.
    stop_value : float
        Value that indicates the end of the sequence.

    Returns
    -------
    ak.Array
        Array of variable length sequences.
    """

    tensor_np = tensor.unsqueeze(-1).detach().cpu().numpy()
    mask = np.zeros((tensor.size(0), tensor.size(1)))
    # find where the stop value is in the sequences
    stop_indices = (tensor_np[:, :, 0] == stop_value).argmax(axis=1)
    # Correct indices where the stop value might not be present (argmax returns 0 in such cases)
    no_stop_value = ~(tensor_np[:, :, 0] == stop_value).any(axis=1)
    stop_indices[no_stop_value] = tensor_np.shape[
        1
    ]  # Use the sequence length for sequences without the stop value

    # build the mask
    for idx, sequence in enumerate(tensor_np):
        mask[idx, : stop_indices[idx]] = 1

    # convert to awkward array
    ak_arr = np_to_ak(tensor_np, names=["dummy"], mask=mask)

    return ak_arr["dummy"]


def arctanh_with_delta(x, delta=1e-5):
    """Arctanh with a small delta to avoid dealing with infinities. This is useful for using
    arctanh as a preprocessing function.

    Parameters
    ----------
    x : ak.Array or np.ndarray
        Input value.
    delta : float, optional
        Small delta to avoid infinities. Default is 1e-5.
    """
    return np.arctanh(ak_clip(x, clip_min=-1 + delta, clip_max=1 - delta))


def fix_padded_logits(logits, mask, factor=1e6):
    """Used to fix a tensor of logits if the sequences are padded after some token.
    The logits of the padded values are all set to 0, except for the first value, which is set to
    `factor`. This is useful when using the logits to calculate the loss.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of logits. Shape (batch_size, seq_len, n_tokens)
    mask : torch.Tensor
        Mask tensor. Shape (batch_size, seq_len)
    factor : float, optional
        Value to set the first token of the padded values to. Default is 1e6.

    Returns
    -------
    torch.Tensor
        Fixed logits.
    """
    # fix the padded logits
    logits = logits * mask.unsqueeze(dim=-1)
    # set the logits of padded values to [1e6, -1e6, -1e6, ...]
    logits = logits + torch.cat(
        [
            (~mask).unsqueeze(-1) * factor,
            torch.zeros_like(logits[:, :, 1:]),
        ],
        dim=-1,
    )
    return logits
