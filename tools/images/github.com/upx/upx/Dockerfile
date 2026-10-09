ARG IMAGE="ubuntu:24.04"
FROM $IMAGE AS builder

WORKDIR "/tmp"

RUN apt-get update && \
    apt-get install -y git cmake g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# the upx release to build and its commit
# renovate: datasource=github-releases depName=upx/upx extractVersion=^v(?<version>.+)$
ARG UPX_VERSION="5.2.1"
# renovate: datasource=git-refs depName=https://github.com/upx/upx currentValue=v${UPX_VERSION}
ARG UPX_COMMIT="b4bdabf638ab6e3715a292182a9c613a4716d7c5"

# fetch the pinned commit, verify it, and fetch the submodule commits it records
RUN git init -q upx && \
    git -C upx fetch -q --depth 1 https://github.com/upx/upx.git "$UPX_COMMIT" && \
    git -C upx checkout -q --detach FETCH_HEAD && \
    test "$(git -C upx rev-parse HEAD)" = "$UPX_COMMIT" && \
    git -C upx submodule update -q --init --recursive --depth 1

# The result uses different compression algorithms than the official release at https://github.com/upx/upx/releases/
# but seems compatible
RUN cd upx && \
    mkdir -p build/release && cd build/release && \
    cmake ../.. && cd ../.. && \
    make

FROM $IMAGE

WORKDIR "/app"

COPY --from=builder /tmp/upx/build/release/upx /usr/local/bin/

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]