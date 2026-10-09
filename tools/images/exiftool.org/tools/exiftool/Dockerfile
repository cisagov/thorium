ARG IMAGE="python:3.11-slim"
FROM $IMAGE

# fail a RUN step if any command in a pipeline fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR "/app"

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN EXIFTOOL_VERSION="$(wget -qO- https://exiftool.org/ver.txt)" && \
    EXIFTOOL_ARCHIVE="Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz" && \
    wget -q "https://downloads.sourceforge.net/project/exiftool/${EXIFTOOL_ARCHIVE}" && \
    CHECKSUM="$(wget -qO- https://exiftool.org/checksums.txt | grep "SHA1(${EXIFTOOL_ARCHIVE}" | awk -F'= ' '{print $2}')" && \
    echo "${CHECKSUM}  ${EXIFTOOL_ARCHIVE}" | /usr/bin/sha1sum -c --quiet - && \
    tar xzf "${EXIFTOOL_ARCHIVE}" && \
    mv Image-ExifTool-*/ /app/exiftool && \
    rm -f "${EXIFTOOL_ARCHIVE}"

COPY run_exiftool.py /app/.

ENTRYPOINT ["python3", "/app/run_exiftool.py"]
