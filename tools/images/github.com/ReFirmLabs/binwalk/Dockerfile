ARG IMAGE="rust:bookworm"
FROM $IMAGE

WORKDIR /app

RUN cargo version

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y p7zip-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cargo install binwalk

COPY wrapper.bash /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.bash"]
