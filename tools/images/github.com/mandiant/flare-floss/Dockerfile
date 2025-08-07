ARG IMAGE="ubuntu:22.04"
FROM $IMAGE AS clone 

WORKDIR "/tmp"

RUN apt update && \
    apt install -y unzip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Download floss zip and add it to the container at /floss
ADD https://github.com/mandiant/flare-floss/releases/download/v3.1.1/floss-v3.1.1-linux.zip /tmp/

# Unzip the floss archive
RUN unzip /tmp/floss-v3.1.1-linux.zip

FROM $IMAGE

WORKDIR "/app"

COPY --from=clone /tmp/floss /app/floss
COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]