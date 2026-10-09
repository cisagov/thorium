ARG IMAGE="ubuntu:24.04"
FROM $IMAGE AS clone 

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR "/tmp"

RUN apt-get update && apt-get install -y unzip

# the quantumstrand release (published in the flare-floss repo) and the sha256 of its linux zip
# renovate: datasource=github-releases depName=mandiant/flare-floss extractVersion=^quantumstrand-(?<version>.+)$
ARG QUANTUMSTRAND_VERSION=beta3
ARG QUANTUMSTRAND_SHA256=38762936c7f9f39969f54fa6c58a669e6bee688799aee93228546605ad0bc9fd

ADD https://github.com/mandiant/flare-floss/releases/download/quantumstrand-${QUANTUMSTRAND_VERSION}/quantumstrand-quantumstrand-${QUANTUMSTRAND_VERSION}-linux.zip /tmp/

# Verify the quantumstrand archive against its pinned sha256 and unzip it
RUN echo "${QUANTUMSTRAND_SHA256}  /tmp/quantumstrand-quantumstrand-${QUANTUMSTRAND_VERSION}-linux.zip" | sha256sum -c - && \
    unzip "/tmp/quantumstrand-quantumstrand-${QUANTUMSTRAND_VERSION}-linux.zip"

FROM $IMAGE

WORKDIR "/app"

COPY --from=clone /tmp/quantumstrand /app/quantumstrand
COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]