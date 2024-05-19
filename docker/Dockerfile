FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    vim \
    neovim \
    wget \
    curl \
    git \
    zsh \
    libxcb-xinerama0 \
    lsb-release \
    bat \
    ripgrep

RUN curl -sL https://raw.githubusercontent.com/wimpysworld/deb-get/main/deb-get | bash -s install deb-get
RUN deb-get install \
    git-delta

# Install GitHub cli
RUN mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# add the 'slurm' user in order to make slurm work from within container
# (if the corresponding libraries are mounted to the container)
RUN adduser --disabled-password --gecos "" slurm

# allow pip install of "sklearn", which should be replaced by "scikit-learn"
# (if this is not used, the github CI pipeline fails from time to time since
# some packages still have "sklearn" in their dependencies)
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY requirements.txt .

RUN pip install -r requirements.txt

SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -y jupyter
RUN git clone https://github.com/minyoungg/vqtorch && cd vqtorch && pip install -e .

# move anaconda binary path to the end, otherwise the "clear" command in the
# terminal is broken
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/anaconda/bin

# activate conda environment by default
RUN ["/bin/bash", "-c", "source /opt/conda/bin/activate"]
