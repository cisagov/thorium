ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

RUN apt-get update \
    && apt-get install -y binutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY wrapper.bash /app/.

# set command
ENTRYPOINT ["bash", "/app/wrapper.bash"]
