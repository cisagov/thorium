FROM thorium-registry.sandia.gov/base/manylinux_2_28_x86_64:latest

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
        libvirt-devel
# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installer && \
      sh installer --default-toolchain nightly-2025-08-01 -y
ENV PATH="/root/.cargo/bin:$PATH"
ENV RUSTFLAGS="-C target-feature=+aes,+sse2"
# install maturin
RUN pipx install maturin[patchelf]
ENV PATH="/root/.local/bin:$PATH"
RUN python3.10 -m pip install --no-cache-dir --user pipx && \
      python3.10 -m pipx ensurepath --global && \
      pipx install maturin
# grab the Thorium binaries
COPY ./ thorium
# fetch our dependencies
RUN cd thorium && \
      cargo fetch

ENTRYPOINT [ "maturin" ]
