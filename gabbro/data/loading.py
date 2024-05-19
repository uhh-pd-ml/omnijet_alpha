import logging
from pathlib import Path

import awkward as ak
import fastjet as fj
import numpy as np
import uproot
import vector

from gabbro.utils.jet_types import get_jet_type_from_file_prefix, jet_types_dict

logger = logging.getLogger(__name__)

vector.register_awkward()


def read_tokenized_jetclass_file(
    filepath,
    particle_features=["part_token_id"],
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
    remove_start_token=False,
    remove_end_token=False,
    shift_tokens_minus_one=False,
    n_load=None,
    random_seed=None,
):
    """Reads a file that contains the tokenized JetClass jets.

    Parameters:
    ----------
    filepath : str
        Path to the file.
    particle_features : List[str], optional
        A list of particle-level features to be loaded. Should only contain "part_token_id".
    labels : List[str], optional
        A list of truth labels to be loaded.
    remove_start_token : bool, optional
        Whether to remove the start token from the tokenized sequence.
    remove_end_token : bool, optional
        Whether to remove the end token from the tokenized sequence.
    shift_tokens_minus_one : bool, optional
        Whether to shift the token values by -1.
    n_load : int, optional
        Number of events to load. If None, all events are loaded.
    random_seed : int, optional
        Random seed for shuffling the data. If None, no shuffling is performed.


    Returns:
    -------
    tokens : List[str]
        A list of file paths.
    """

    ak_tokens = ak.from_parquet(filepath)

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(ak_tokens))
        ak_tokens = ak_tokens[permutation]

    if n_load is not None:
        ak_tokens = ak_tokens[:n_load]
    # extract jet type from filename and create the corresponding labels
    jet_type_prefix = filepath.split("/")[-1].split("_")[0] + "_"
    jet_type_name = get_jet_type_from_file_prefix(jet_type_prefix)
    # one-hot encode the jet type
    labels_onehot = ak.Array(
        {
            f"label_{jet_type}": np.ones(len(ak_tokens)) * (jet_type_name == jet_type)
            for jet_type in jet_types_dict
        }
    )
    if remove_start_token:
        ak_tokens = ak_tokens[:, 1:]
    if remove_end_token:
        ak_tokens = ak_tokens[:, :-1]
    if shift_tokens_minus_one:
        ak_tokens = ak_tokens - 1

    x_ak = ak.Array({"part_token_id": ak_tokens})

    return x_ak, labels_onehot[labels]


def read_jetclass_file(
    filepath,
    particle_features=["part_pt", "part_eta", "part_phi", "part_energy"],
    jet_features=["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
    return_p4=False,
    n_load=None,
):
    """Loads a single file from the JetClass dataset.

    Parameters:
    ----------
    filepath : str
        Path to the ROOT data file.
    particle_features : List[str], optional
        A list of particle-level features to be loaded.
        Possible options are:
        - part_px
        - part_py
        - part_pz
        - part_energy
        - part_deta
        - part_dphi
        - part_d0val
        - part_d0err
        - part_dzval
        - part_dzerr
        - part_charge
        - part_isChargedHadron
        - part_isNeutralHadron
        - part_isPhoton
        - part_isElectron
        - part_isMuon

    jet_features : List[str], optional
        A list of jet-level features to be loaded.
        Possible options are:
        - jet_pt
        - jet_eta
        - jet_phi
        - jet_energy
        - jet_nparticles
        - jet_sdmass
        - jet_tau1
        - jet_tau2
        - jet_tau3
        - jet_tau4
        - aux_genpart_eta
        - aux_genpart_phi
        - aux_genpart_pid
        - aux_genpart_pt
        - aux_truth_match

    labels : List[str], optional
        A list of truth labels to be loaded.
        - label_QCD
        - label_Hbb
        - label_Hcc
        - label_Hgg
        - label_H4q
        - label_Hqql
        - label_Zqq
        - label_Wqq
        - label_Tbqq
        - label_Tbl

    return_p4 : bool, optional
        Whether to return the 4-momentum of the particles.

    Returns:
    -------
    x_particles : ak.Array
        An awkward array of the particle-level features.
    x_jets : ak.Array
        An awkward array of the jet-level features.
    y : ak.Array
        An awkward array of the truth labels (one-hot encoded).
    p4 : ak.Array, optional
        An awkward array of the 4-momenta of the particles.
    """

    if n_load is not None:
        table = uproot.open(filepath)["tree"].arrays()[:n_load]
    else:
        table = uproot.open(filepath)["tree"].arrays()

    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
            # massless particles -> this changes the result slightly,
            # i.e. for example top jets then have a mass of 171.2 instead of 172
            # "mass": ak.zeros_like(table["part_px"]),
        }
    )
    p4_jet = ak.sum(p4, axis=1)

    p4_massless = ak.zip(
        {
            "pt": p4.pt,
            "eta": p4.eta,
            "phi": p4.phi,
            "mass": ak.zeros_like(table["part_px"]),
        },
        with_name="Momentum4D",
    )
    p4_jet_massless = ak.sum(p4_massless, axis=1)

    table["part_pt"] = p4.pt
    table["part_eta"] = p4.eta
    table["part_phi"] = p4.phi
    table["part_ptrel"] = table["part_pt"] / p4_jet.pt
    table["part_erel"] = table["part_energy"] / p4_jet.energy
    table["part_etarel"] = p4.deltaeta(p4_jet)
    table["part_phirel"] = p4.deltaphi(p4_jet)
    table["part_deltaR"] = p4.deltaR(p4_jet)
    table["part_energy_raw"] = table[
        "part_energy"
    ]  # workaround to have this feature twice if we want

    table["part_pt_massless"] = p4_massless.pt
    table["part_px_massless"] = p4_massless.px
    table["part_py_massless"] = p4_massless.py
    table["part_pz_massless"] = p4_massless.pz
    table["part_ptrel_massless"] = p4_massless.pt / p4_jet_massless.pt
    table["part_energy_massless"] = p4_massless.energy
    table["part_erel_massless"] = p4_massless.energy / p4_jet_massless.energy
    table["part_etarel_massless"] = p4_massless.deltaeta(p4_jet_massless)
    table["part_phirel_massless"] = p4_massless.deltaphi(p4_jet_massless)
    table["part_deltaR_massless"] = p4_massless.deltaR(p4_jet_massless)
    table["part_energy_raw_massless"] = p4_massless.energy

    x_particles = table[particle_features] if particle_features is not None else None
    x_jets = table[jet_features] if jet_features is not None else None
    y = ak.values_astype(table[labels], "int32") if labels is not None else None

    if return_p4:
        return x_particles, x_jets, y, p4

    return x_particles, x_jets, y


def load_particles(
    filenames,
    mode="initial",
    n_subjets=10,
    n_load=None,
    particle_features=None,
):
    """Load particles from root files and return them as awkward array.

    Parameters
    ----------
    filenames : list of str
        List of filenames to load.
    mode : str, optional
        Mode of loading. Can be one of
        - "initial": load the initial particles
        - "subjets": cluster the jets into subjets and return them
        - "all_intermediate": cluster the jets using a huge radius and return all intermediate jets
    n_subjets : int, optional
        Number of subjets to cluster the jets into. Only used if mode="subjets".
    n_load : int, optional
        Number of events to load. If None, all events are loaded.
    particle_features : list of str, optional
        List of particle-level features to load.

    Returns
    -------
    ak_particle_features : awkward array
        Array of particle features.
    ak_particle_p4s : awkward array
        Array of particles 4-momenta.
    labels : array
        Array of labels indicating the jet type.
    """

    if (particle_features is not None and particle_features != []) and mode != "initial":
        raise ValueError(
            "particle_features can only be used in mode='initial', not in mode='subjets' "
            " or mode='all_intermediate'\n"
            f"particle_features={particle_features}, mode={mode}"
        )

    for i, filename in enumerate(filenames):
        logger.info(f"Loading particles from {filename}")

        ak_particle_features_i, _, _, ak_particles_p4s_i = read_jetclass_file(
            filename,
            return_p4=True,
            particle_features=particle_features,
        )

        if n_load is not None:
            ak_particles_p4s_i = ak_particles_p4s_i[:n_load]
            if particle_features is not None:
                ak_particle_features_i = ak_particle_features_i[:n_load]

        if mode == "initial":
            pass
        elif mode == "subjets":
            print(f"Clustering into {n_subjets} subjets")
            # remove jets with less than n_subjets constituents
            mask_subjets = ak.num(ak_particles_p4s_i) >= n_subjets
            ak_particles_p4s_i = ak_particles_p4s_i[mask_subjets]

            # cluster jets into subjets
            jetdef = fj.JetDefinition(fj.kt_algorithm, 8)
            print("Clustering jets with fastjet")
            print("Jet definition:", jetdef)
            cluster = fj.ClusterSequence(ak_particles_p4s_i, jetdef)
            ak_particles_p4s_i = cluster.exclusive_jets(n_jets=n_subjets)

            ak_particles_p4s_i = ak.zip(
                {
                    "pt": ak_particles_p4s_i.pt,
                    "eta": ak_particles_p4s_i.eta,
                    "phi": ak_particles_p4s_i.phi,
                    "mass": ak_particles_p4s_i.mass,
                },
                with_name="Momentum4D",
            )
        elif mode == "all_intermediate":
            # cluster jets using huge radius
            jetdef = fj.JetDefinition(fj.antikt_algorithm, 8)
            print("Clustering jets with fastjet")
            print("Jet definition:", jetdef)
            cluster = fj.ClusterSequence(ak_particles_p4s_i, jetdef)
            ak_particles_p4s_i = cluster.jets()

            ak_particles_p4s_i = ak.zip(
                {
                    "pt": ak_particles_p4s_i.pt,
                    "eta": ak_particles_p4s_i.eta,
                    "phi": ak_particles_p4s_i.phi,
                    "mass": ak_particles_p4s_i.mass,
                },
                with_name="Momentum4D",
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # correct negative masses
        mass_eps = ak.ones_like(ak_particles_p4s_i.pt) * 1e-5
        ak_particles_p4s_i = ak.zip(
            {
                "pt": ak_particles_p4s_i.pt,
                "eta": ak_particles_p4s_i.eta,
                "phi": ak_particles_p4s_i.phi,
                "mass": ak.where(ak_particles_p4s_i.mass <= 0, mass_eps, ak_particles_p4s_i.mass),
            },
            with_name="Momentum4D",
        )

        # get the label indicating the jet type
        filename_last = Path(filename).name
        jet_type_label = None
        for jet_type, jet_type_dict in jet_types_dict.items():
            prefix = jet_type_dict["file_prefix"]
            if prefix in filename_last:
                if jet_type_label is not None:
                    raise ValueError(f"Found multiple jet types in {filename}")
                jet_type_label = jet_type_dict["label"]

        labels_i = np.ones(len(ak_particles_p4s_i)) * jet_type_label

        if i == 0:
            ak_particles_p4s = ak_particles_p4s_i
            ak_particle_features = (
                ak_particle_features_i if particle_features is not None else None
            )
            np_labels = labels_i
        else:
            ak_particles_p4s = ak.concatenate([ak_particles_p4s, ak_particles_p4s_i], axis=0)
            if particle_features is not None:
                ak_particle_features = ak.concatenate(
                    [ak_particle_features, ak_particle_features_i], axis=0
                )
            np_labels = np.concatenate([np_labels, labels_i], axis=0)
    return ak_particle_features, ak_particles_p4s, np_labels
