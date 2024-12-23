FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# # Install linux dependencies
RUN apt-get update \
 && apt-get install -y libssl-dev software-properties-common git sqlite3 zip curl rsync sagemath wget git gcc

WORKDIR /project
RUN git clone https://github.com/unitn-sml/wmi-pa.git \
 && cd /project/wmi-pa \
 && pip3 install .

# Install mathsat
RUN pysmt-install --msat --confirm-agreement
RUN wmipa-install --nra

WORKDIR /wmipa

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


