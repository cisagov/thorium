FROM ubuntu:22.04

WORKDIR /bin
COPY ./target/x86_64-unknown-linux-musl/release/thorctl .

# run as an unprivileged user with a numeric id, so Kubernetes can verify runAsNonRoot; the
# mounted /conf and /keys files must be readable by it
RUN groupadd --system --gid 10001 thorium && \
    useradd --system --uid 10001 --gid thorium --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin thorium
USER 10001:10001

ENTRYPOINT ["./thorctl", "--admin", "/conf/thorium.yml", "--keys", "/keys/keys.yml"]
