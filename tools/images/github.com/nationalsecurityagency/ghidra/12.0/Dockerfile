ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# Download and install Ghidra
ENV GHIDRA_VERSION=12.0.3
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_12.0.3_build/ghidra_12.0.3_PUBLIC_20260210.zip
ENV GHIDRA_SHA=90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981

RUN apt update \
    && apt install -y --no-install-recommends curl unzip openjdk-21-jdk-headless \
    && curl -L -k -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
