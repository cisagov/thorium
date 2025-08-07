ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y python3 clamav && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Grab definitions
RUN freshclam
 
COPY run_clamav.py /app/.

ENTRYPOINT ["python3", "/app/run_clamav.py"]