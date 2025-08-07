ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y build-essential && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN gcc XORStrings.c -o xorstrings

ENTRYPOINT ["/bin/sh", "/app/wrapper.sh"]
