ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tshark && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.bash /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.bash"]