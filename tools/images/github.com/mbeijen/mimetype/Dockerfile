ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

RUN apt update && \
    apt install -y libfile-mimeinfo-perl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY wrapper.bash /app/.

ENTRYPOINT ["bash", "/app/wrapper.bash"]