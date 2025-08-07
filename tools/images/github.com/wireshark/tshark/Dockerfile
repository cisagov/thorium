ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y tshark && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.bash /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.bash"]