# syntax=docker/dockerfile:1

ARG IMAGE="ubuntu:24.04"
# renovate: datasource=github-releases depName=VirusTotal/yara extractVersion=^v(?<version>.+)$
ARG YARA_VERSION="4.5.8"
# the sha256 of the GitHub source archive for the YARA_VERSION tag
ARG YARA_SHA256="c322414975ff6f701149856613afdcd92a7e6939c284c798ae3c85618197efaa"

############################
# Stage 1: Build YARA
############################
FROM ${IMAGE} AS yara-builder

ARG YARA_VERSION
ARG YARA_SHA256

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      automake \
      libtool \
      make \
      gcc \
      pkg-config \
      flex \
      bison \
      openssl \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/yara-build

RUN curl -fsSL \
      "https://github.com/VirusTotal/yara/archive/refs/tags/v${YARA_VERSION}.tar.gz" \
      -o yara.tar.gz && \
    echo "${YARA_SHA256}  yara.tar.gz" | sha256sum -c - && \
    tar -xzf yara.tar.gz --strip-components=1 && \
    rm yara.tar.gz && \
    ./bootstrap.sh && \
    ./configure \
      --prefix=/usr/local \
      --disable-static && \
    make -j"$(nproc)" && \
    make DESTDIR=/tmp/yara-root install && \
    strip /tmp/yara-root/usr/local/bin/yara || true && \
    strip /tmp/yara-root/usr/local/bin/yarac || true && \
    find /tmp/yara-root/usr/local/lib -type f -name "*.so*" -exec strip {} \; || true


############################
# Stage 2: Build and compile rules
############################
FROM ${IMAGE} AS rules-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      python3 \
      libssl3 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=yara-builder /tmp/yara-root/ /

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/yara.conf && \
    ldconfig && \
    yara --version && \
    yarac --version

WORKDIR /app

COPY . .

RUN mkdir -p /app/rules/yara-forge /app/rules/custom && \
    curl -fsSL "https://github.com/YARAHQ/yara-forge/releases/latest/download/yara-forge-rules-extended.zip" -o /tmp/yara-forge-rules-extended.zip && \
    unzip -oj /tmp/yara-forge-rules-extended.zip "*.yar" -d /app/rules/yara-forge && \
    rm -f /tmp/yara-forge-rules-extended.zip && \
    python3 /app/validate_rules.py && \
    mapfile -d '' rule_files < <( \
      find /app/rules/yara-forge /app/rules/custom \
        -maxdepth 1 \
        -type f \
        \( -iname '*.yar' -o -iname '*.yara' \) \
        -print0 | sort -z \
    ) && \
    if (( ${#rule_files[@]} == 0 )); then \
      echo "[ERROR] No .yar or .yara files found for compilation." >&2; \
      exit 1; \
    fi && \
    yarac "${rule_files[@]}" /app/rules.bin && \
    yara -C /app/rules.bin /bin/bash && \
    rm -rf /app/rules


############################
# Stage 3: Runtime image
############################
FROM ${IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      libssl3 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=yara-builder /tmp/yara-root/ /
COPY --from=rules-builder /app /app

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/yara.conf && \
    ldconfig && \
    yara --version && \
    test -f /app/rules.bin

WORKDIR /app

ENTRYPOINT ["python3", "/app/run_yara.py"]