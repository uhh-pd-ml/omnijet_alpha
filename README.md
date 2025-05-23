# OmniJet-α: The first cross-task foundation model for particle physics

<div align="center">

Joschka Birk, Anna Hallin, Gregor Kasieczka

[![arXiv](https://img.shields.io/badge/arXiv-2403.05618-b31b1b.svg)](https://arxiv.org/abs/2403.05618)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This repository contains the code for the results presented in the paper
'[OmniJet-α: The first cross-task foundation model for particle physics
](https://arxiv.org/abs/2403.05618)'.

The code was also used in:

[![arXiv](https://img.shields.io/badge/arXiv-2412.10504-b31b1b.svg)](https://arxiv.org/abs/2412.10504)
[![arXiv](https://img.shields.io/badge/arXiv-2503.19165-b31b1b.svg)](https://arxiv.org/abs/2503.19165)

**Abstract**:

> Foundation models are multi-dataset and multi-task machine learning methods that once pre-trained can be fine-tuned for a large variety of downstream applications. The successful development of such general-purpose models for physics data would be a major breakthrough as they could improve the achievable physics performance while at the same time drastically reduce the required amount of training time and data.
> We report significant progress on this challenge on several fronts. First, a comprehensive set of evaluation methods is introduced to judge the quality of an encoding from physics data into a representation suitable for the autoregressive generation of particle jets with transformer architectures (the common backbone of foundation models). These measures motivate the choice of a higher-fidelity tokenization compared to previous works. Finally, we demonstrate transfer learning between an unsupervised problem (jet generation) and a classic supervised task (jet tagging) with our new OmniJet-α model. This is the first successful transfer between two different and actively studied classes of tasks and constitutes a major step in the building of foundation models for particle physics.

<img src=assets/OmniJet_different_stages_overview.excalidraw.png width=900 style="border-radius:10px">

______________________________________________________________________

**Table of contents**:

- [OmniJet-α: The first cross-task foundation model for particle physics](#omnijet-%CE%B1-the-first-cross-task-foundation-model-for-particle-physics)
  - [How to run the code](#how-to-run-the-code)
    - [Dataset](#dataset)
    - [Installation](#installation)
    - [Tokenization](#tokenization)
    - [Generative training](#generative-training)
    - [Transfer learning / Classifier training](#transfer-learning--classifier-training)
  - [Citation](#citation)

## How to run the code

### Dataset

Instructions on how to download the dataset can be found in the
repository [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).

### Installation

The recommended (and by us tested) way of running the code is to use the
provided docker image at
[`jobirk/omnijet` on DockerHub](https://hub.docker.com/repository/docker/jobirk/omnijet/general).
The requirements listed in `docker/requirements.txt` are installed in the `conda` environment
`base` of the base image (official pytorch image).
Thus, you have to make sure that the `conda` environment is activated when running the code,
which can be done with `source /opt/conda/bin/activate`.

An interactive session inside a container can be started by running the following command:

```shell
# on a machine with Singularity
singularity shell docker://jobirk/omnijet:latest  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
#
# on a machine with Docker
docker run -it --rm jobirk/omnijet:latest bash  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
```

Alternatively, you can install the requirements from the `docker/requirements.txt` file, but
you'll have to add `pytorch` to the list of requirements, since this is not
included in the `requirements.txt` file (we use the official pytorch image as
base image).

Furthermore, you'll have to add/create a `.env` file in the root of the project
with the following content:

```bash
JETCLASS_DIR="<path to the jetclass dataset i.e. where the train_100M, val_5M, .. folders are>"
JETCLASS_DIR_TOKENIZED="<path to where you want to save the tokenized jetclass dataset>"

# stuff for hydra
LOG_DIR="<path to log dir>"
COMET_API_TOKEN="<your comet api token>"
HYDRA_FULL_ERROR=1
```

### Tokenization

To play around with the already-trained VQ-VAE model, you can download the
checkpoint (see `checkpoints/README.md` for instructions) and then have
a look at the notebook
[`examples/notebooks/example_tokenize_and_reconstruct_jets.ipynb`](https://github.com/uhh-pd-ml/omnijet_alpha/blob/main/examples/notebooks/example_tokenize_and_reconstruct_jets.ipynb).

You can run the training of the VQ-VAE model by running the following command:

```bash
python gabbro/train.py experiment=example_experiment_tokenization
```

You can then evaluate a tokenization model by running the following command:

```bash
python scripts/evaluate_tokenization_ckpt.py --n_eval=100000 --ckpt_path=<path to the checkpoint>
```

The result of the evaluation will appear in a subfolder of the run directory.

To create the tokenized dataset, you can run the following command:

```bash
python scripts/create_tokenized_dataset.py --ckpt_path=<path to the checkpoint> --n_files_train=100 --n_files_val=5 --n_files_test=5
```

Make sure to adjust the `--n_files_*` arguments to your needs, and set the env variable
`JETCLASS_DIR` and `JETCLASS_DIR_TOKENIZED` in the `.env` file.

Afterwards, the tokenized dataset will be saved in a subdirectory of the
`JETCLASS_DIR_TOKENIZED` directory and can be used to train the backbone model.

### Generative training

To play around with the already-trained generative model, you can download the
checkpoint (see `checkpoints/README.md` for instructions) and then have
a look at the notebook
[`examples/notebooks/example_generate_jets.ipynb`](https://github.com/uhh-pd-ml/omnijet_alpha/blob/main/examples/notebooks/example_generate_jets.ipynb).

If you want to run a generative training, you first have to create the tokenized
dataset (see above).
Note that you have to make sure that the checkpoint of the tokenizer is saved/copied
to that directory as `model_ckpt.ckpt` and the training config as `config.yaml`
(this is necessary since the gen. training will look for those files to reconstruct
tokens back to physical space).

You can then run the training of the generative model by running the following command:

```bash
python gabbro/train.py experiment=example_experiment_generative
```

### Transfer learning / Classifier training

You can run the training of the classifier model by running the following command:

```bash
python gabbro/train.py experiment=example_experiment_classification
```

You can then evaluate a classifier model by running the following command:

```bash
python scripts/evaluate_classication_ckpt.py --ckpt_path=<path to the checkpoint>
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Birk:2024knn,
    author = "Birk, Joschka and Hallin, Anna and Kasieczka, Gregor",
    title = "{OmniJet-\ensuremath{\alpha}: the first cross-task foundation model for particle physics}",
    eprint = "2403.05618",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1088/2632-2153/ad66ad",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "5",
    number = "3",
    pages = "035031",
    year = "2024"
}
```
