ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libfile-mimeinfo-perl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.bash /app/.

ENTRYPOINT ["bash", "/app/wrapper.bash"]