FROM python:3.8

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends 
    #\
    #python3.8 python3-pip python3-setuptools python3-dev

WORKDIR /src

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.2.0-cp38-cp38-manylinux2010_x86_64.whl

COPY . /src

#CMD jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root