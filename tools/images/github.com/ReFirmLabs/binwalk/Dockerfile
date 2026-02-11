ARG IMAGE="rust:bullseye"
FROM $IMAGE

WORKDIR /app

RUN cargo version

RUN apt clean all && \
    apt update && \
    apt install -y p7zip-full && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN cargo install binwalk

COPY wrapper.bash /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.bash"]
