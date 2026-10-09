ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Download and install Ghidra
# renovate: datasource=github-releases depName=NationalSecurityAgency/ghidra extractVersion=^Ghidra_(?<version>.+)_build$ allowedVersions=12.0.x
ENV GHIDRA_VERSION=12.0.4
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_12.0.4_build/ghidra_12.0.4_PUBLIC_20260303.zip
ENV GHIDRA_SHA=c3b458661d69e26e203d739c0c82d143cc8a4a29d9e571f099c2cf4bda62a120

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip openjdk-21-jdk-headless \
    && curl -L -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
