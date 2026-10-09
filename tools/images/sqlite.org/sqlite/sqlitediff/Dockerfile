ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

# copy run script into container
COPY . /app

# install sqlite utility
RUN apt-get update && \
    apt-get install -y sqlite3 sqlite3-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# set command
ENTRYPOINT ["bash", "/app/run.sh"]
