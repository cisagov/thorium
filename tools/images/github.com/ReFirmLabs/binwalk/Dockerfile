ARG IMAGE="rust:bullseye"
FROM $IMAGE

WORKDIR /app

RUN cargo install binwalk

COPY wrapper.bash /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.bash"]
