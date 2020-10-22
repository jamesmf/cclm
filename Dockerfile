FROM tensorflow/tensorflow:2.3.1-gpu

WORKDIR /app

COPY . /app/cclm

CMD pip install -e cclm