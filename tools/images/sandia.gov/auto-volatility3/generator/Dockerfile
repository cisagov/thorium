ARG IMAGE="python:3.11-slim"
FROM "rust:bookworm" AS builder

WORKDIR "/tmp"

# copy in generator source and cargo config
COPY . /tmp/.

# build generator binary
ENV RUSTFLAGS="-C target-feature=+aes,+sse2"
RUN git clone https://github.com/cisagov/thorium.git thorium
RUN rustup toolchain install nightly --profile minimal
RUN rustup default nightly
RUN cargo build --release

FROM $IMAGE

# install volatility3
RUN pip3 install --no-cache-dir volatility3 --break-system-packages

# make sure our symbols directory is writable by all users
RUN chmod -R 777 /usr/local/lib/python3.*/dist-packages/volatility3 || true
RUN chmod -R 777 /usr/lib/python3.*/dist-packages/volatility3 || true

# add our tool binary from build image
COPY --from=builder /tmp/target/release/auto-volatility3 /usr/bin/.

RUN apt-get update && \
    apt-get install -y zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# work out of /app
WORKDIR /app

ENTRYPOINT ["auto-volatility3", "--keys", "/opt/thorium-keys/keys.yml"]
