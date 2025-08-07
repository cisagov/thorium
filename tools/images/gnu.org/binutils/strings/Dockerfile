ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

RUN apt update \
    && apt install -y binutils \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]