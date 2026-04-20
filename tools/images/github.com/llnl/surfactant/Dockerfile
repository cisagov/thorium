ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

# git: required for the fuzzyhashes plugin's git+https install
# build-essential: the plugin pulls py-tlsh, which has no wheel and compiles C++
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### REPO: https://github.com/LLNL/Surfactant

# install surfactant and plugins
RUN pip install --upgrade pip setuptools wheel && \
    pip install surfactant && \
    pip install git+https://github.com/LLNL/Surfactant#subdirectory=plugins/fuzzyhashes

ADD wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]
