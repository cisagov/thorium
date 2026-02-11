ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"
COPY . /app

# Add trufflehog binary
ADD https://github.com/trufflesecurity/trufflehog/releases/download/v3.87.1/trufflehog_3.87.1_linux_amd64.tar.gz /app
RUN tar -xvf trufflehog_3.87.1_linux_amd64.tar.gz; \
    rm trufflehog_3.87.1_linux_amd64.tar.gz

ENTRYPOINT ["/bin/bash", "/app/wrapper.sh"]