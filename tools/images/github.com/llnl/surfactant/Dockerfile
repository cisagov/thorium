ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

RUN apt update && \
    apt install -y python3 python3-pip git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

### REPO: https://github.com/LLNL/Surfactant

# install surfactant and plugins
RUN pip install surfactant && \
    pip install git+https://github.com/LLNL/Surfactant#subdirectory=plugins/fuzzyhashes

ADD wrapper.sh /app/.

# setup our workdir
WORKDIR /app

ENTRYPOINT ["bash", "/app/wrapper.sh"]