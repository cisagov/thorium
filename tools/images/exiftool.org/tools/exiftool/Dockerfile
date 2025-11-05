ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y curl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN EXIFTOOL_VERSION=`curl -s https://exiftool.org/ver.txt` && \
    EXIFTOOL_ARCHIVE=Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz && \
    curl -s -O https://exiftool.org/$EXIFTOOL_ARCHIVE && \
    CHECKSUM=`curl -s https://exiftool.org/checksums.txt | grep SHA1\(${EXIFTOOL_ARCHIVE} | awk -F'= ' '{print $2}'` && \
    echo "${CHECKSUM}  ${EXIFTOOL_ARCHIVE}" | /usr/bin/sha1sum -c --quiet - && \
    tar xzf $EXIFTOOL_ARCHIVE && \
    mv Image-ExifTool-*/ /app/exiftool && \
    rm -f $EXIFTOOL_ARCHIVE

COPY run_exiftool.py /app/.

ENTRYPOINT ["python3", "/app/run_exiftool.py"]
