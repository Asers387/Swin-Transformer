FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
