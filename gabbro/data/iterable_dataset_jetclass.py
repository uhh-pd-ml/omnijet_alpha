import glob
import logging
import random
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import torch
import vector
from torch.utils.data import DataLoader, IterableDataset

from gabbro.data.data_tokenization import reconstruct_jetclass_file
from gabbro.data.loading import read_jetclass_file, read_tokenized_jetclass_file
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    calc_additional_kinematic_features,
)
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

vector.register_awkward()


class CustomIterableDataset(IterableDataset):
    """Custom IterableDataset that loads data from multiple files."""

    def __init__(
        self,
        files_dict: dict,
        n_files_at_once: int = None,
        n_jets_per_file: int = None,
        max_n_files_per_type: int = None,
        shuffle_files: bool = True,
        shuffle_data: bool = True,
        seed: int = 4697,
        seed_shuffle_data: int = 3838,
        pad_length: int = 128,
        logger_name: str = "CustomIterableDataset",
        feature_dict: dict = None,
        labels_to_load: list = None,
        token_reco_cfg: dict = None,
        token_id_cfg: dict = None,
        load_only_once: bool = False,
        shuffle_only_once: bool = False,
        random_seed_for_per_file_shuffling: int = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        files_dict : dict
            Dict with the file names for each type. Can be e.g. a dict like
            {"tbqq": ["tbqq_0.root", ...], "qcd": ["qcd_0.root", ...], ...}.
        n_files_at_once : int, optional
            Number of files to load at once. If None, one file per files_dict key
            is loaded.
        n_jets_per_file : int, optional
            Number of jets loaded from each individual file. Defaults to None, which
            means that all jets are loaded.
        max_n_files_per_type : int, optional
            Maximum number of files to use per type. If None, all files are used.
            Can be used to use e.g. always the first file from the sorted list of files
            in validation.
        shuffle_files : bool, optional
            Whether to shuffle the list of files.
        shuffle_data : bool, optional
            Whether to shuffle the data after loading.
        seed : int, optional
            Random seed.
        seed_shuffle_data : int, optional
            Random seed for shuffling the data. This is useful if you want to shuffle
            the data in the same way for different datasets (e.g. train and val).
            The default value is 3838.
        pad_length : int, optional
            Maximum number of particles per jet. If a jet has more particles, the
            first pad_length particles are used, the rest is discarded.
        logger_name : str, optional
            Name of the logger.
        feature_dict : dict, optional
            Dictionary with the features to load. The keys are the names of the features
            and the values are the preprocessing parameters passed to the
            `ak_select_and_preprocess` function.
        labels_to_load : list, optional
            List with the jet_type labels to load.
        token_reco_cfg : dict, optional
            Dictionary with the configuration to reconstruct the tokenized jetclass files.
            If None, this is not used.
        token_id_cfg : dict, optional
            Dictionary with the tokenization configuration, this is to be used when the
            token-id data is to be loaded. If None, this is ignored.
        load_only_once : bool, optional
            If True, the data is loaded only once and then returned in the same order
            in each iteration. NOTE: this is only useful if the whole dataset fits into
            memory. If the dataset is too large, this will lead to a memory error.
        shuffle_only_once : bool, optional
            If True, the data is shuffled only once and then returned in the same order
            in each iteration. NOTE: this should only be used for val/test.
        random_seed_for_per_file_shuffling : int, optional
            Random seed for shuffling the jets within a file. This is useful if you want
            to only load a subset of the jets from a file and want to choose different
            jets in different training runs.
            If load_only_once is False, this is ignored.

        **kwargs
            Additional keyword arguments.

        """
        if feature_dict is None:
            raise ValueError("feature_dict must be provided.")
        if labels_to_load is None:
            raise ValueError("labels_to_load must be provided.")
        self.logger = logging.getLogger(logger_name)
        self.logger.info(f"Using seed {seed}")
        self.pad_length = pad_length
        self.shuffle_data = shuffle_data
        self.shuffle_files = shuffle_files
        self.max_n_files_per_type = max_n_files_per_type
        self.n_jets_per_file = n_jets_per_file
        self.feature_dict = feature_dict
        self.labels_to_load = labels_to_load
        self.particle_features_list = [feat for feat in self.feature_dict.keys() if "part" in feat]
        self.seed_shuffle_data = seed_shuffle_data
        self.load_only_once = load_only_once
        self.shuffle_only_once = shuffle_only_once
        self.data_shuffled = False
        self.random_seed_for_per_file_shuffling = random_seed_for_per_file_shuffling

        if self.random_seed_for_per_file_shuffling is not None:
            if not self.load_only_once:
                self.logger.warning(
                    "random_seed_for_per_file_shuffling is only used if load_only_once is True."
                )
                self.random_seed_for_per_file_shuffling = None
            else:
                self.logger.info(
                    f"Using random seed {self.random_seed_for_per_file_shuffling} for per-file shuffling."
                )

        self.logger.info(f"Using the following labels: {self.labels_to_load}")
        self.logger.info(f"Using the following particle features: {self.particle_features_list}")
        self.logger.info(f"pad_length {self.pad_length} for the number of particles per jet.")
        self.logger.info(f"shuffle_data={self.shuffle_data}")
        self.logger.info(f"shuffle_files={self.shuffle_files}")
        self.logger.info(
            "Number of jets loaded per file: "
            f"{self.n_jets_per_file if self.n_jets_per_file is not None else 'all'}"
        )
        self.logger.info("Using the following features:")
        for feat, params in self.feature_dict.items():
            self.logger.info(f"- {feat}: {params}")
        self.files_dict = {}
        for jet_type, files in files_dict.items():
            expanded_files = []
            for file in files:
                expanded_files.extend(sorted(list(glob.glob(file))))
            self.files_dict[jet_type] = (
                expanded_files
                if max_n_files_per_type is None
                else expanded_files[:max_n_files_per_type]
            )

            self.logger.info(f"Files for jet_type {jet_type}:")
            for file in self.files_dict[jet_type]:
                self.logger.info(f" - {file}")

        if self.load_only_once:
            logger.warning(
                "load_only_once is True. This means that there will only be the initial data loading."
            )

        # add all files from the dict to a list (the values are lists of files)
        self.file_list = []
        for files in self.files_dict.values():
            self.file_list.extend(files)

        # if not specified how many files to use at once, use one file per jet_type
        if n_files_at_once is None:
            self.n_files_at_once = len(self.files_dict)
        else:
            if n_files_at_once > len(self.file_list):
                self.logger.warning(
                    f"n_files_at_once={n_files_at_once} is larger than the number of files in the"
                    f" dataset ({len(self.file_list)})."
                )
                self.logger.warning(f"Setting n_files_at_once to {len(self.file_list)}.")
                self.n_files_at_once = len(self.file_list)
            else:
                self.n_files_at_once = n_files_at_once

        self.logger.info(f"Will load {self.n_files_at_once} files at a time and combine them.")

        self.file_indices = np.array([0, self.n_files_at_once])
        self.file_iterations = len(self.file_list) // self.n_files_at_once
        if self.load_only_once:
            self.file_iterations = 1

        self.current_part_data = None
        self.current_part_mask = None
        self.token_reco_cfg = token_reco_cfg
        self.token_id_cfg = token_id_cfg

    def get_data(self):
        """Returns a generator (i.e. iterator) that goes over the current files list and returns
        batches of the corresponding data."""
        # Iterate over jet_type
        self.logger.debug("\n>>> __iter__ called\n")
        self.file_indices = np.array([0, self.n_files_at_once])

        # shuffle the file list
        if self.shuffle_files:
            self.logger.info(">>> Shuffling files")
            random.shuffle(self.file_list)
            # self.logger.info(">>> self.file_list:")
            # for filename in self.file_list:
            #     self.logger.info(f" - {filename}")

        # Iterate over files
        for j in range(self.file_iterations):
            self.logger.debug(20 * "-")
            # Increment file index if not first iteration
            if j > 0:
                self.logger.info(">>> Incrementing file index")
                self.file_indices += self.n_files_at_once

            # stop the iteration if self.file_indices[1] is larger than the number of files
            # FIXME: this means that the last batch of files (in case the number of files is not
            # divisible by self.n_files_at_once) is not used --> fix this
            # but if shuffling is used, this should not be a problem
            if self.file_indices[1] <= len(self.file_list):
                self.load_next_files()

                # loop over the current data
                for i in range(len(self.current_part_data)):
                    yield {
                        "part_features": self.current_part_data[i],
                        "part_mask": self.current_part_mask[i],
                        "jet_type_labels_one_hot": self.current_jet_type_labels_one_hot[i],
                        "jet_type_labels": torch.argmax(self.current_jet_type_labels_one_hot[i]),
                    }

    def __iter__(self):
        """returns an iterable which represents an iterator that iterates over the dataset."""
        return iter(self.get_data())

    def load_next_files(self):
        if self.load_only_once:
            if self.current_part_data is not None:
                self.logger.warning("Data has already been loaded. Will not load again.")
                self.shuffle_current_data()
                return
        self.part_data_list = []
        self.mask_data_list = []
        self.jet_type_labels_list = []

        self.current_files = self.file_list[self.file_indices[0] : self.file_indices[1]]
        self.logger.info(f">>> Loading next files - self.file_indices={self.file_indices}")
        if self.load_only_once:
            self.logger.warning("Loading data only once. Will not load again.")
            self.logger.warning("--> This will be the data for all iterations.")
        for i_file, filename in enumerate(self.current_files):
            self.logger.info(f"{i_file + 1} / {len(self.current_files)} : {filename}")

            if self.token_reco_cfg is not None:
                gpu_available = torch.cuda.is_available()
                _, ak_x_particles, ak_jet_type_labels = reconstruct_jetclass_file(
                    filename_in=filename,
                    model_ckpt_path=self.token_reco_cfg["ckpt_file"],
                    config_path=self.token_reco_cfg["config_file"],
                    start_token_included=self.token_reco_cfg["start_token_included"],
                    end_token_included=self.token_reco_cfg["end_token_included"],
                    shift_tokens_by_minus_one=self.token_reco_cfg["shift_tokens_by_minus_one"],
                    device="cuda" if gpu_available else "cpu",
                    return_labels=True,
                )
                self.logger.info("Calculating additional kinematic features.")
                ak_x_particles = calc_additional_kinematic_features(ak_x_particles)

            elif self.token_id_cfg is not None:
                ak_x_particles, ak_jet_type_labels = read_tokenized_jetclass_file(
                    filename,
                    labels=self.labels_to_load,
                    remove_start_token=self.token_id_cfg.get("remove_start_token", False),
                    remove_end_token=self.token_id_cfg.get("remove_end_token", False),
                    shift_tokens_minus_one=self.token_id_cfg.get("shift_tokens_minus_one", False),
                    n_load=self.n_jets_per_file,
                    random_seed=self.random_seed_for_per_file_shuffling,
                )
                ak_x_particles = ak.Array(
                    {
                        "part_token_id": ak_x_particles["part_token_id"],
                        "part_token_id_without_last": ak_x_particles["part_token_id"][:, :-1],
                        "part_token_id_without_first": ak_x_particles["part_token_id"][:, 1:],
                    }
                )
            else:
                # read the data from the file
                # can add jet features, labels, and p4s here
                ak_x_particles, _, ak_jet_type_labels = read_jetclass_file(
                    filename,
                    particle_features=self.particle_features_list,
                    labels=self.labels_to_load,
                    n_load=self.n_jets_per_file,
                )

            ak_x_particles = ak_select_and_preprocess(ak_x_particles, self.feature_dict)
            ak_x_particles_padded, ak_mask_particles = ak_pad(
                ak_x_particles, self.pad_length, return_mask=True
            )
            np_x_particles_padded = ak_to_np_stack(
                ak_x_particles_padded, names=self.particle_features_list
            )
            np_mask_particles = ak.to_numpy(ak_mask_particles)
            np_jet_type_labels = ak_to_np_stack(ak_jet_type_labels, names=self.labels_to_load)

            # add the data to the lists
            self.part_data_list.append(torch.tensor(np_x_particles_padded))
            self.mask_data_list.append(torch.tensor(np_mask_particles, dtype=torch.bool))
            self.jet_type_labels_list.append(torch.tensor(np_jet_type_labels))

        # concatenate the data from all files
        self.current_part_data = torch.cat(self.part_data_list, dim=0)
        self.current_part_mask = torch.cat(self.mask_data_list, dim=0)
        self.current_jet_type_labels_one_hot = torch.cat(self.jet_type_labels_list, dim=0)

        self.shuffle_current_data()

        self.logger.info(
            f">>> Data loaded. (self.current_part_data.shape = {self.current_part_data.shape})"
        )

    def shuffle_current_data(self):
        # shuffle the data
        if self.shuffle_only_once and self.data_shuffled:
            self.logger.info("Data has already been shuffled. Will not shuffle again.")
            return
        if self.shuffle_data:
            rng = np.random.default_rng()
            if self.seed_shuffle_data is not None:
                self.logger.info(f"Shuffling data with seed {self.seed_shuffle_data}")
                rng = np.random.default_rng(self.seed_shuffle_data)
            perm = rng.permutation(len(self.current_part_data))
            self.current_part_data = self.current_part_data[perm]
            self.current_part_mask = self.current_part_mask[perm]
            self.current_jet_type_labels_one_hot = self.current_jet_type_labels_one_hot[perm]
            self.data_shuffled = True


class IterableDatamodule(L.LightningDataModule):
    def __init__(
        self,
        dataset_kwargs_train: dict,
        dataset_kwargs_val: dict,
        dataset_kwargs_test: dict,
        dataset_kwargs_common: dict,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__()

        # save the parameters as attributes
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Prepare the data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_train,
                **self.hparams.dataset_kwargs_common,
            )
            self.val_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_val,
                **self.hparams.dataset_kwargs_common,
            )
        elif stage == "test":
            self.test_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_test,
                **self.hparams.dataset_kwargs_common,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
