ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

# copy run script into container
COPY . /app

# install sqlite utility
RUN apt update && \
    apt install -y sqlite3 sqlite3-tools && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# set command
ENTRYPOINT ["bash", "/app/run.sh"]
