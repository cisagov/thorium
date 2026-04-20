ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR /app

RUN apt update && \
    apt install -y tcpflow && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]