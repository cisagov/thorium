FROM thorium-registry.sandia.gov/base/ubuntu:latest
ARG TOKEN

# update apt and install curl
RUN apt install curl -y
# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installer && \
      sh installer --default-toolchain nightly -y
# install build dependencies
RUN apt update && apt install build-essential libssl-dev pkg-config libclang-dev clang-15 musl-tools cmake libvirt-dev jq -y
# add cargo to PATH and install rust nightly and musl builds
ENV PATH=${PATH}:/root/.cargo/bin
RUN rustup default nightly && \
      rustup target add x86_64-unknown-linux-musl
# download and install redis for integration testing
RUN apt install wget -y  && \
      curl -L --output redis-6.2.4.tar.gz https://download.redis.io/releases/redis-6.2.4.tar.gz && \
      tar xzf redis-6.2.4.tar.gz && \
      cd redis-6.2.4 && \
      make && \
      make install
# download and install scylla for integration testing
RUN wget https://s3.amazonaws.com/downloads.scylladb.com/deb/scylladb.gpg.pubkey && \
      gpg --homedir /tmp --no-default-keyring --keyring /etc/apt/keyrings/scylladb.gpg --keyserver hkp://keyserver.ubuntu.com:80 --keyserver-options http-proxy=http://thorium:pass@proxy.sandia.gov --recv-keys a43e06657bac99e3 && \
      wget -O /etc/apt/sources.list.d/scylla.list http://downloads.scylladb.com/deb/debian/scylla-6.2.list && \
      apt-get update && \
      apt-get install -y scylla && \
      rm scylladb.gpg.pubkey
# set the locales and setup sculla
ENV LC_ALL="en_US.UTF-8"
RUN apt install locales && \
      locale-gen en_US.UTF-8 && \
      dpkg-reconfigure locales && \
      scylla_dev_mode_setup --developer-mode 1
# download crane
RUN mkdir app && \
      cd app && \
      VERSION=$(curl -s "https://api.github.com/repos/google/go-containerregistry/releases/latest" | jq -r '.tag_name') && \
      export OS=Linux && \
      export ARCH=x86_64 && \
      curl -sL "https://github.com/google/go-containerregistry/releases/download/${VERSION}/go-containerregistry_${OS}_${ARCH}.tar.gz" > go-containerregistry.tar.gz && \
      tar xzvf go-containerregistry.tar.gz && \
      rm gcrane krane go-containerregistry.tar.gz LICENSE README.md && \
      cd ..
# grab the Thorium binaries
COPY ./ thorium
# fetch our dependencies
RUN cd thorium && \
      cargo fetch
