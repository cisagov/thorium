FROM ubuntu:24.04

# install utilities
RUN apt update -y && \
      apt install -y lsof curl libssl-dev jq

# copy the api in
WORKDIR /app
ADD ./target/release/thorium thorium
ADD ./target/release/thorium-operator thorium-operator
ADD ./target/release/thoradm thoradm
ADD ./target/release/thorium-scaler thorium-scaler
ADD ./target/release/thorium-search-streamer thorium-search-streamer
ADD ./target/release/thorium-event-handler thorium-event-handler
# Add UI bundle to root path
ADD ./ui/dist ui
# copy the user and developer docs in
ADD ./api/docs/book docs/user
ADD ./target/doc docs/dev
# copy our binaries in
ADD ./target/x86_64-unknown-linux-musl/release/thorctl binaries/linux/x86-64/thorctl
ADD ./target/x86_64-unknown-linux-musl/release/thorium-agent binaries/linux/x86-64/thorium-agent
ADD ./target/x86_64-unknown-linux-musl/release/thorium-reactor binaries/linux/x86-64/thorium-reactor
ADD ./target/x86_64-unknown-linux-musl/release/thoradm binaries/linux/x86-64/thoradm
ADD ./target/x86_64-unknown-linux-musl/release/thorium-operator binaries/linux/x86-64/thorium-operator
# copy windows binaries to target paths
#ADD ./target/x86_64-pc-windows-gnu/release/thorctl.exe binaries/windows/x86-64/thorctl.exe
#ADD ./target/x86_64-pc-windows-gnu/release/thorium-agent.exe binaries/windows/x86-64/thorium-agent.exe
#ADD ./target/x86_64-pc-windows-gnu/release/thorium-reactor.exe binaries/windows/x86-64/thorium-reactor.exe
# copy macOS binaries to target paths
ADD ./target/x86_64-apple-darwin/release/thorctl binaries/darwin/x86-64/thorctl
ADD ./target/aarch64-apple-darwin/release/thorctl binaries/darwin/arm64/thorctl
# copy arm binaries to target paths
#ADD ./target/aarch64-unknown-linux-musl/release/thorctl binaries/linux/aarch64/thorctl
# copy the thorctl install script to the right path
ADD ./api/docs/src/scripts/install-thorctl.sh binaries/install-thorctl.sh

# make glibc compiled binaries executable
RUN chmod +x thorium \
  thoradm \
  thorium-operator \
  thorium-scaler \
  thorium-search-streamer \
  thorium-event-handler

# make cross compiled binaries executable
RUN chmod -R +x binaries

# add banner to default path
ADD ./ui/src/assets/banner.txt banner.txt
# download crane
RUN VERSION=$(curl -s "https://api.github.com/repos/google/go-containerregistry/releases/latest" | jq -r '.tag_name') && \
      export OS=Linux && \
      export ARCH=x86_64 && \
      curl -sL "https://github.com/google/go-containerregistry/releases/download/${VERSION}/go-containerregistry_${OS}_${ARCH}.tar.gz" > go-containerregistry.tar.gz && \
      tar xzvf go-containerregistry.tar.gz && \
      rm gcrane krane go-containerregistry.tar.gz LICENSE README.md

ENTRYPOINT ["./thorium-operator", "operate"]
