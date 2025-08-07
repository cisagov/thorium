ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

# install bulk_extractor deps
RUN apt update && \
    apt install -y git python3 autoconf make build-essential flex libexpat1-dev \
    libxml2-utils libewf-dev libssl-dev libtool make pkg-config zlib1g-dev libre2-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# clone the bulk_extractor project
RUN git clone --recurse-submodules https://github.com/simsong/bulk_extractor.git  /app/bulk_extractor

# configure and install bulk_extractor
RUN cd /app/bulk_extractor && \
    git checkout tags/v2.1.1 && \
    sh ./bootstrap.sh && \
    ./configure && \
    make -j32 && \
    make install

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]