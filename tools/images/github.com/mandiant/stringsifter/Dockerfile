ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

# install python > version 3.9
RUN apt update && \
    apt install -y software-properties-common build-essential gcc \
                   python3 python3-pip python3-dev python3-distutils -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    python3 -m pip install stringsifter

COPY wrapper.sh /app/.

ENTRYPOINT ["/bin/bash", "/app/wrapper.sh"]