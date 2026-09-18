ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# Download and install Ghidra
ENV GHIDRA_VERSION=10.3.2
ENV GHIDRA_URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.3.2_build/ghidra_10.3.2_PUBLIC_20230711.zip
ENV GHIDRA_SHA=a658677a87d0be12ab65bd7962f471875b81a2dd2ea35d69cc3201555ca1bd6f

RUN apt update \
    && apt install -y --no-install-recommends curl unzip openjdk-17-jdk-headless \
    && curl -L -k -o /tmp/ghidra.zip "${GHIDRA_URL}" \
    && echo "${GHIDRA_SHA} /tmp/ghidra.zip" | sha256sum -c - \
    && unzip -q -d /tmp/ /tmp/ghidra.zip \
    && mv "/tmp/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra \
    && rm /tmp/ghidra.zip

ENTRYPOINT ["/opt/ghidra/support/analyzeHeadless"]
