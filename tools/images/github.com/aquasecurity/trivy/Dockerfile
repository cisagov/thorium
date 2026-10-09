ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The Trivy release to install and the sha256 of its .deb packages, taken from the release's
# trivy_<version>_checksums.txt (Sigstore-signed by the aquasecurity/trivy release workflow).
# Bump these together, and only to a release whose checksums and signature have been verified.
# renovate: datasource=github-releases depName=aquasecurity/trivy extractVersion=^v(?<version>.+)$
ARG TRIVY_VERSION=0.75.0
ARG TRIVY_SHA256_AMD64=b6d7b944b6cbf4e7de5b4bc6c3635900c50fed53b64b77b99a5fbc3136cd1d5d
ARG TRIVY_SHA256_ARM64=23018f1f7136b8391bb300e2579bcf4759e3c64012856f51c0fb6fffe1ee3841

# download the pinned Trivy .deb for this architecture, verify its checksum, and install it
RUN apt-get update -y && \
  apt-get install -y ca-certificates wget && \
  case "$(dpkg --print-architecture)" in \
    amd64) TRIVY_ARCH=64bit; TRIVY_SHA256="${TRIVY_SHA256_AMD64}" ;; \
    arm64) TRIVY_ARCH=ARM64; TRIVY_SHA256="${TRIVY_SHA256_ARM64}" ;; \
    *) echo "unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
  esac && \
  wget -q -O /tmp/trivy.deb "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-${TRIVY_ARCH}.deb" && \
  echo "${TRIVY_SHA256}  /tmp/trivy.deb" | sha256sum -c - && \
  dpkg -i /tmp/trivy.deb && \
  rm /tmp/trivy.deb && \
  trivy --version | grep -x "Version: ${TRIVY_VERSION}" && \
  rm -rf /var/lib/apt/lists/*

# prefetch and save our DB (wrapper.sh moves /tmp/cache/* back into /root/.cache/ at run time,
# so keep the trivy/ cache folder itself under /tmp/cache)
RUN trivy fs --scanners vuln,secret,misconfig,license /tmp && \
    mkdir -p /tmp/cache && \
    mv /root/.cache/* /tmp/cache/ && \
    chmod -R 777 /tmp/cache

# Add our wrapper.sh script
WORKDIR /app
COPY wrapper.sh .

# setup our entrypoint (wrapper.sh is not executable in git, so run it with bash)
ENTRYPOINT ["bash", "/app/wrapper.sh"]
