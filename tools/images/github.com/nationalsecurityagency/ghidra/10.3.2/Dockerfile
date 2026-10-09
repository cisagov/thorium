ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Download and install Ghidra
# renovate: datasource=github-releases depName=NationalSecurityAgency/ghidra extractVersion=^Ghidra_(?<version>.+)_build$ allowedVersions=10.3.x
ENV GHIDRA_VERSION=10.3.3
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.3.3_build/ghidra_10.3.3_PUBLIC_20230829.zip
ENV GHIDRA_SHA=63833361bea8ef5ada1bc28cd2aa2ae4ab43204d2672b595500372582152eebe

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip openjdk-17-jdk-headless \
    && curl -L -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
