ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

RUN apt update && \
    apt install -y file && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]