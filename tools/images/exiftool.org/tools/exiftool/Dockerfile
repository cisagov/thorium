ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y python3 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://exiftool.org/Image-ExifTool-13.38.tar.gz /app/

RUN tar -xzvf Image-ExifTool-13.38.tar.gz && \
    rm Image-ExifTool-13.38.tar.gz

COPY run_exiftool.py /app/.

ENTRYPOINT ["python3", "/app/run_exiftool.py"]
