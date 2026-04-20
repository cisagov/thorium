ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

# lightgbm (a stringsifter dependency) ships a native library that links against
# the GNU OpenMP runtime; libgomp1 is not present in the -slim base image
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# stringsifter pins numpy==1.24.4, which only has wheels up through Python 3.11;
# on 3.11 the whole dependency tree resolves to prebuilt wheels (no toolchain)
RUN pip install --upgrade pip setuptools wheel && \
    pip install stringsifter

COPY wrapper.sh /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.sh"]
