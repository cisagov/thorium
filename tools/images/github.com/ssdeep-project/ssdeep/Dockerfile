ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt clean all && \
    apt update && \
    apt install ssdeep -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.sh /app/.

ENTRYPOINT ["/bin/bash", "wrapper.sh"]