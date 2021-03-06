FROM ubuntu:16.04

# Environment
ENV DEBIAN_FRONTEND noninteractive

# Install all basic requirements
RUN \
    apt-get update -q -y && \
    apt-get -y dist-upgrade && \
    apt-get -y --no-install-recommends  install \
        build-essential \
        wget \
        unzip \
        gfortran \
        # BLAS
        libatlas-base-dev \
        # Python 2
        python-setuptools \
        python-pip \
        python-dev \
		cmake \
		git \
		openjdk-8-jdk \
		maven \
    && \
    apt-get clean && \
    rm -rf /var/cache/apt/*

RUN git clone https://github.com/tlindener/xgboost --recursive && \
	cd xgboost && \
	chmod +x build.sh && \
	./build.sh
RUN cd xgboost/jvm-packages && \ 
	mvn package