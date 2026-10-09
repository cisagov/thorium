FROM thorium-registry.sandia.gov/base/manylinux_2_28_x86_64:latest

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# install build dependencies
RUN yum -y update && \
    yum -y install \
        curl \
        openssl-devel \
        pkgconfig \
        clang \
        clang-devel \
        cmake \
        perl-core \
        libvirt-devel && \
    yum clean all && \
    rm -rf /var/cache/yum
# the rustup release to install and the sha256 of its x86_64 Linux rustup-init from static.rust-lang.org
# renovate: datasource=github-tags depName=rust-lang/rustup
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf -o rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" && \
      echo "${RUSTUP_INIT_SHA256}  rustup-init" | sha256sum -c - && \
      chmod +x rustup-init && \
      ./rustup-init --default-toolchain nightly -y && \
      rm rustup-init
ENV PATH="/root/.cargo/bin:$PATH"
ENV RUSTFLAGS="-C target-feature=+aes,+sse2"
# install maturin (PyO3 build system)
RUN pipx install maturin[patchelf]
ENV PATH="/root/.local/bin:$PATH"
RUN python3.10 -m pip install --no-cache-dir --user pipx && \
      python3.10 -m pipx ensurepath --global && \
      pipx install maturin
# install twine (for uploading PyPi packages)
RUN python3.10 -m pip install --no-cache-dir --upgrade twine
# grab the Thorium binaries
WORKDIR /thorium
COPY ./ .
# fetch our dependencies
RUN cargo fetch

ENTRYPOINT [ "maturin" ]
