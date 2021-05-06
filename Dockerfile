FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && \
    apt-get install -y git

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && source $HOME/.poetry/env

COPY . /app/cclm

WORKDIR /app/cclm

RUN $HOME/.poetry/bin/poetry install