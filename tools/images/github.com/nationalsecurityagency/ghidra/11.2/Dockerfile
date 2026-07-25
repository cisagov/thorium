ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# Download and install Ghidra
ENV GHIDRA_VERSION=11.2
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2_build/ghidra_11.2_PUBLIC_20240926.zip
ENV GHIDRA_SHA=a98fe01038fe8791c54b121ede545ea799d26358794d7c2ac09fa3f5054f3cdc

RUN apt update \
    && apt install -y --no-install-recommends curl unzip openjdk-21-jdk-headless \
    && curl -L -k -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
