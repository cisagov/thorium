ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# trufflehog shells out to git to clone and scan repos in git mode (wrapper.sh),
# and needs the CA bundle to verify found secrets over HTTPS
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR "/app"
COPY . /app

# the trufflehog release to install and the sha256 of its linux_amd64 tarball, from the
# release's trufflehog_<version>_checksums.txt
# renovate: datasource=github-releases depName=trufflesecurity/trufflehog extractVersion=^v(?<version>.+)$
ARG TRUFFLEHOG_VERSION=3.99.0
ARG TRUFFLEHOG_SHA256=e35ae3fd1e4b7a0c1699d0fda1814282bf906d03379bd4337b092b3a84c6740b

# Add trufflehog binary
ADD https://github.com/trufflesecurity/trufflehog/releases/download/v${TRUFFLEHOG_VERSION}/trufflehog_${TRUFFLEHOG_VERSION}_linux_amd64.tar.gz /app
# Verify the archive against its pinned sha256 and extract it
RUN echo "${TRUFFLEHOG_SHA256}  trufflehog_${TRUFFLEHOG_VERSION}_linux_amd64.tar.gz" | sha256sum -c - && \
    tar -xvf "trufflehog_${TRUFFLEHOG_VERSION}_linux_amd64.tar.gz" && \
    rm "trufflehog_${TRUFFLEHOG_VERSION}_linux_amd64.tar.gz"

ENTRYPOINT ["/bin/bash", "/app/wrapper.sh"]