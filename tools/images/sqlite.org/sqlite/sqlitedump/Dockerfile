# Debian builds sqlite3 with SQLITE_ENABLE_DBPAGE_VTAB, which .dbinfo needs (Ubuntu does not)
ARG IMAGE="debian:trixie"
FROM $IMAGE

WORKDIR "/app"

# copy run script into container
COPY run.sh /app/run.sh

# install sqlite utility
RUN apt-get update && \
    apt-get install -y sqlite3 jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# set command
ENTRYPOINT ["bash", "/app/run.sh"]
