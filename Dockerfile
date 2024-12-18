FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# # Install linux dependencies
RUN apt-get update \
 && apt-get install -y libssl-dev software-properties-common git sqlite3 zip curl rsync sagemath wget git gcc

WORKDIR /project
RUN git clone https://github.com/unitn-sml/wmi-pa.git \
 && cd /project/wmi-pa \
 && pip3 install .

 # Install Latte
RUN apt-get update \
&& apt-get install -y  build-essential m4 cmake g++ make 

RUN wget -c "https://github.com/latte-int/latte/releases/download/version_1_7_5/latte-integrale-1.7.5.tar.gz" \
 && tar xvf latte-integrale-1.7.5.tar.gz \
 && cd latte-integrale-1.7.5\
 && ./configure --prefix=/project/latte --with-default=/project/latte \
 && make \
 && make install

ENV PATH="$PATH:/project/latte/bin"


# Install Volesti
RUN apt-get install -y lp-solve \
&& wmipa-install --volesti -yf \
&& echo "export PATH=/root/.wmipa/approximate-integration/bin:$PATH" >> ~/.bash_profile

# Install mathsat
RUN pysmt-install --msat --confirm-agreement
RUN wmipa-install --nra
