ARG IMAGE="rust:bullseye"
FROM $IMAGE AS builder

WORKDIR "/tmp"

# copy in generator source and cargo config
COPY . /tmp/.

# build generator binary
ENV RUSTFLAGS="-C target-cpu=native"
RUN rustup toolchain install nightly --profile minimal
RUN rustup default nightly
RUN cargo build --release

FROM $IMAGE

# add our tool binary from build image
COPY --from=builder /tmp/target/release/auto-volatility3 /usr/bin/.

# work out of /app
WORKDIR /app

# add our rules
ADD rules rules/.

ENTRYPOINT ["auto-volatility3", "--keys", "/opt/thorium-keys/keys.yml"]