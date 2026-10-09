ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN gcc XORStrings.c -o xorstrings

ENTRYPOINT ["/bin/sh", "/app/wrapper.sh"]
