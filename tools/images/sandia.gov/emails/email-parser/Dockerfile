ARG IMAGE="rust:bullseye"
FROM $IMAGE AS builder

WORKDIR "/tmp"

COPY . /tmp/.

RUN cargo build --release

FROM $IMAGE

COPY --from=builder /tmp/target/release/email-parser /usr/bin/.

ENTRYPOINT ["email-parser"]