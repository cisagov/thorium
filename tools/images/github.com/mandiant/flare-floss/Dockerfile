ARG IMAGE="ubuntu:24.04"
FROM $IMAGE AS clone 

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR "/tmp"

RUN apt-get update && \
    apt-get install -y unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# the floss release to install and the sha256 of its linux zip
# renovate: datasource=github-releases depName=mandiant/flare-floss extractVersion=^v(?<version>\d.+)$
ARG FLOSS_VERSION=3.1.1
ARG FLOSS_SHA256=40c05a869f34f7e2417b17ca290cc54bd3671ee1f0a2d9bd5103284c01a54666

# Download floss zip and add it to the container at /floss
ADD https://github.com/mandiant/flare-floss/releases/download/v${FLOSS_VERSION}/floss-v${FLOSS_VERSION}-linux.zip /tmp/

# Verify the floss archive against its pinned sha256 and unzip it
RUN echo "${FLOSS_SHA256}  /tmp/floss-v${FLOSS_VERSION}-linux.zip" | sha256sum -c - && \
    unzip "/tmp/floss-v${FLOSS_VERSION}-linux.zip"

FROM $IMAGE

WORKDIR "/app"

COPY --from=clone /tmp/floss /app/floss
COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]