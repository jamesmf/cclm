FROM tensorflow/tensorflow:2.3.1-gpu

COPY ./setup.py  /app/cclm/
COPY ./cclm/__init__.py /app/cclm/cclm/

RUN pip install -e /app/cclm/

WORKDIR /app

COPY . /app/cclm

RUN pip install -e cclm/