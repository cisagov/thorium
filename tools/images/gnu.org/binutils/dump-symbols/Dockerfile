ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

RUN apt update \
    && apt install -y binutils \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY wrapper.bash /app/.

# set command
ENTRYPOINT ["bash", "/app/wrapper.bash"]
