ARG IMAGE="python:3.11-slim"
FROM $IMAGE

# install volatility3
RUN pip3 install --no-cache-dir volatility3

# make sure our symbols directory is writable by all users
RUN chmod -R 777 /usr/local/lib/python3.*/dist-packages/volatility3 || true
RUN chmod -R 777 /usr/lib/python3.*/dist-packages/volatility3 || true

RUN apt-get update && \
    apt-get install -y zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# add our wrapper
COPY wrapper.sh /app/wrapper.sh

ENTRYPOINT ["bash", "/app/wrapper.sh"]
