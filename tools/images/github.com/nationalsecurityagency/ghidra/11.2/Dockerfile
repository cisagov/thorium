ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Download and install Ghidra (analysis loads the JDK font manager, which needs fontconfig, a font and libharfbuzz0b)
# renovate: datasource=github-releases depName=NationalSecurityAgency/ghidra extractVersion=^Ghidra_(?<version>.+)_build$ allowedVersions=11.2.x
ENV GHIDRA_VERSION=11.2.1
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2.1_build/ghidra_11.2.1_PUBLIC_20241105.zip
ENV GHIDRA_SHA=ce4db5117da0fbaf8f33863fec4f40902f754f06b68945a59fb1c0f9b1bc461c

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip openjdk-21-jdk-headless fontconfig libharfbuzz0b \
    && curl -L -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
