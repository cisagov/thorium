FROM ubuntu:24.04

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# install utilities
RUN apt-get update -y && \
      apt-get install -y lsof curl libssl-dev jq && \
      rm -rf /var/lib/apt/lists/*

# copy the api in
WORKDIR /app
COPY ./target/release/thorium-api thorium-api
COPY ./target/release/thorium-operator thorium-operator
COPY ./target/release/thoradm thoradm
COPY ./target/release/thorium-scaler thorium-scaler
COPY ./target/release/thorium-search-streamer thorium-search-streamer
COPY ./target/release/thorium-event-handler thorium-event-handler
# Add UI bundle to root path
COPY ./ui/dist ui
# copy their user and developer docs in
COPY ./api/docs/book docs/user
COPY ./target/doc docs/dev
# copy our binaries in
COPY ./target/x86_64-unknown-linux-musl/release/thorctl binaries/linux/x86-64/thorctl
COPY ./target/x86_64-unknown-linux-musl/release/thorium-agent binaries/linux/x86-64/thorium-agent
COPY ./target/x86_64-unknown-linux-musl/release/thorium-reactor binaries/linux/x86-64/thorium-reactor
COPY ./target/x86_64-unknown-linux-musl/release/thoradm binaries/linux/x86-64/thoradm
COPY ./target/x86_64-unknown-linux-musl/release/thorium-operator binaries/linux/x86-64/thorium-operator
# copy windows binaries to target paths
COPY ./target/x86_64-pc-windows-gnu/release/thorctl.exe binaries/windows/x86-64/thorctl.exe
COPY ./target/x86_64-pc-windows-gnu/release/thorium-agent.exe binaries/windows/x86-64/thorium-agent.exe
COPY ./target/x86_64-pc-windows-gnu/release/thorium-reactor.exe binaries/windows/x86-64/thorium-reactor.exe
# copy macos binaries to target paths
COPY ./target/x86_64-apple-darwin/release/thorctl binaries/darwin/x86-64/thorctl
COPY ./target/aarch64-apple-darwin/release/thorctl binaries/darwin/arm64/thorctl
# copy the thorctl install script to the right path
COPY ./api/docs/src/scripts/install-thorctl.sh binaries/install-thorctl.sh
# add banner to default path
COPY ./ui/src/assets/banner.txt banner.txt

# the crane release to install and the sha256 of its Linux x86_64 archive from the release's checksums.txt
# renovate: datasource=github-releases depName=google/go-containerregistry
ARG CRANE_VERSION=v0.22.1
ARG CRANE_SHA256=0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0

# make glibc compiled and cross compiled binaries executable, then download and verify crane
RUN chmod +x thorium-api \
      thoradm \
      thorium-operator \
      thorium-scaler \
      thorium-search-streamer \
      thorium-event-handler && \
    chmod -R +x binaries && \
    curl -fsSL -o go-containerregistry.tar.gz "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" && \
    echo "${CRANE_SHA256}  go-containerregistry.tar.gz" | sha256sum -c - && \
    tar xzf go-containerregistry.tar.gz crane && \
    rm go-containerregistry.tar.gz

ENTRYPOINT ["./thorium-operator", "operate"]
