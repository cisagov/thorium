ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt-get update && \
    apt-get install -y python3 clamav && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Grab definitions
RUN freshclam
 
COPY run_clamav.py /app/.

ENTRYPOINT ["python3", "/app/run_clamav.py"]