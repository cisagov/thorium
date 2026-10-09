ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt-get clean && \
    apt-get update && \
    apt-get install ssdeep -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.sh /app/.

ENTRYPOINT ["/bin/bash", "wrapper.sh"]