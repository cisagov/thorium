ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR /app

# install bulk_extractor deps
RUN apt-get update && \
    apt-get install -y git python3 autoconf automake make build-essential flex libexpat1-dev \
    libxml2-utils libewf-dev libssl-dev libtool pkg-config zlib1g-dev libre2-dev libabsl-dev \
    libgcrypt20-dev libgpg-error-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# the bulk_extractor release to build and its commit
# renovate: datasource=github-releases depName=simsong/bulk_extractor extractVersion=^v(?<version>.+)$
ARG BULK_EXTRACTOR_VERSION="2.2.0"
# renovate: datasource=git-refs depName=https://github.com/simsong/bulk_extractor currentValue=v${BULK_EXTRACTOR_VERSION}
ARG BULK_EXTRACTOR_COMMIT="9af62917d73e0b631395064aba7ab4de7fbbf696"

# fetch the pinned commit, verify it, and fetch the submodule commits it records
RUN git init -q /app/bulk_extractor && \
    git -C /app/bulk_extractor fetch -q --depth 1 https://github.com/simsong/bulk_extractor.git "$BULK_EXTRACTOR_COMMIT" && \
    git -C /app/bulk_extractor checkout -q --detach FETCH_HEAD && \
    test "$(git -C /app/bulk_extractor rev-parse HEAD)" = "$BULK_EXTRACTOR_COMMIT" && \
    git -C /app/bulk_extractor submodule update -q --init --recursive --depth 1

# configure and install bulk_extractor
RUN cd /app/bulk_extractor && \
    sh ./bootstrap.sh && \
    ./configure && \
    make -j32 && \
    make install

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]