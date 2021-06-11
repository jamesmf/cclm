FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y git wget curl zip

SHELL ["/bin/bash", "-c"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && source $HOME/.poetry/env

COPY . /app/cclm

WORKDIR /app/cclm

# RUN   apt-get update && apt-get upgrade -y && apt-get install -y locales wget git curl zip vim apt-transport-https tzdata language-pack-nb language-pack-nb-base manpages \
#     build-essential libjpeg-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev libreadline6-dev libsqlite3-dev tk-dev libffi-dev libpng-dev libfreetype6-dev helix-cli
# RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
# RUN $HOME/.pyenv/bin/pyenv install 3.8.10

# RUN $HOME/.poetry/bin/poetry install
RUN $HOME/.poetry/bin/poetry config virtualenvs.create false && $HOME/.poetry/bin/poetry install