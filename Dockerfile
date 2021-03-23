FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && \
    apt-get install -y git

COPY ./setup.py  /app/cclm/
COPY ./cclm/__init__.py /app/cclm/cclm/
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt && pip install -e /app/cclm/

WORKDIR /app

COPY . /app/cclm

RUN pip install -e cclm/