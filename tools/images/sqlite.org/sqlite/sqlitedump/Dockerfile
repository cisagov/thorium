ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

# install sqlite utility
RUN apt update && \
    apt install -y sqlite3 jq && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# set command
ENTRYPOINT ["bash", "/app/run.sh"]
