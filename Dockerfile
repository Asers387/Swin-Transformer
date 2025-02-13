FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
