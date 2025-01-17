FROM ubuntu:20.04


ARG DEBIAN_FRONTEND=noninteractive

# # Install linux dependencies
RUN apt-get update \
 && apt-get install -y libssl-dev software-properties-common git sqlite3 zip curl rsync sagemath wget git gcc

RUN apt-get update \
 && apt-get install -y sudo

# Create user with specific UID
RUN useradd -u 31772 -m des
RUN adduser des sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /home/des/app
RUN chown -R des:des /home/des/app 
RUN chmod -R 775 /home/des

# Create and activate virtual environment with Python 3.10
# Switch to des
USER des
WORKDIR /home/des/app

RUN sudo apt install python3.8-venv -y
RUN python3 -m venv /home/des/venv
ENV PATH="/home/des/venv/bin:$PATH"

RUN pip3 install wheel
RUN git clone https://github.com/unitn-sml/wmi-pa.git \
 && cd wmi-pa \
 && pip3 install .

# # Install mathsat
RUN pip3 install pysmt && \
 pysmt-install --msat --confirm-agreement
RUN wmipa-install --nra

 # Install Latte
 RUN sudo apt-get update \
 && sudo apt-get install -y  build-essential m4 cmake g++ make 

RUN wget -c "https://github.com/latte-int/latte/releases/download/version_1_7_5/latte-integrale-1.7.5.tar.gz" \
 && tar xvf latte-integrale-1.7.5.tar.gz \
 && cd latte-integrale-1.7.5\
 && ./configure --prefix=/home/des/app/latte --with-default=/home/des/app/latte \
 && make -j 8 \
 && make install

ENV PATH="$PATH:/home/des/app/latte/bin"

# # Install Volesti
RUN sudo apt-get install -y lp-solve libboost-all-dev \
&& wmipa-install --volesti -yf
ENV PATH="/home/des/.wmipa/approximate-integration/bin:$PATH"

# Install Psi Solver
RUN sudo apt -qq update && sudo apt -qq install -y wget unzip xz-utils  git libxml2-dev curl
RUN sudo apt-get update && sudo apt-get install -y build-essential
# RUN sudo apt-get update && sudo apt-get install -y gnuplot

# Install dlang
RUN curl -fsS https://dlang.org/install.sh | bash -s dmd
RUN echo "source ~/dlang/dmd-2.109.1/activate" >> /home/des/.bashrc

RUN git clone https://github.com/eth-sri/psi.git \
&& cd ./psi \
 && ./dependencies-release.sh \
 && ./build-release.sh \
 && mkdir bin \
 && mv psi ./bin

ENV PATH="/home/des/app/psi/bin:$PATH"


COPY ./ ./wmi-pa-w-volappx
WORKDIR /home/des/app/wmi-pa-w-volappx

RUN pip install -r requirements.txt

WORKDIR /home/des/app/
ENTRYPOINT ["/bin/bash"]