ARG IMAGE="ubuntu:24.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt-get update && \
    apt-get install -y zip fcrackzip p7zip-full p7zip-rar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY dictionary.txt /app/.
COPY wrapper.bash /app/.


ENTRYPOINT ["bash", "/app/wrapper.bash"]
