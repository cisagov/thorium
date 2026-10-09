ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR "/app"
COPY run_die.py /app/.

# the Detect It Easy release to install and the sha256 of its Ubuntu 24.04 package
# renovate: datasource=github-releases depName=horsicq/DIE-engine extractVersion=^(?<version>\d.+)$
ARG DIE_VERSION=3.21
ARG DIE_SHA256=96e8e82e822e3ca8a829536312b92e8b241147fe38c8ab29c9dbfc1ab21aa25d

RUN apt-get update -qq && \
    apt-get install -y wget libglib2.0-0 libqt5sql5 libqt5svg5 libqt5gui5 libqt5widgets5 libqt5opengl5 libqt5dbus5 libqt5scripttools5 libqt5script5 libqt5network5 && \
    wget -q "https://github.com/horsicq/DIE-engine/releases/download/${DIE_VERSION}/die_${DIE_VERSION}_Ubuntu_24.04_amd64.deb" && \
    echo "${DIE_SHA256}  die_${DIE_VERSION}_Ubuntu_24.04_amd64.deb" | sha256sum -c - && \
    dpkg -i "die_${DIE_VERSION}_Ubuntu_24.04_amd64.deb" && \
    rm "die_${DIE_VERSION}_Ubuntu_24.04_amd64.deb" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python3", "/app/run_die.py"]
