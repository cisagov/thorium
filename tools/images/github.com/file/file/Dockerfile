ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR /app

RUN apt-get update && \
    apt-get install -y file && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]