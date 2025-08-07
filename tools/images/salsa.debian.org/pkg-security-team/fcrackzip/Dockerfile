ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y zip fcrackzip p7zip-full p7zip-rar && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY dictionary.txt /app/.
COPY wrapper.bash /app/.


ENTRYPOINT ["bash", "/app/wrapper.bash"]
