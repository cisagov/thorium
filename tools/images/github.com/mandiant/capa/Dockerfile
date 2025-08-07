ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip install flare-capa

RUN apt update && \
    apt install -y git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/mandiant/capa-rules.git /app/capa-rules
RUN git clone --depth 1 https://github.com/mandiant/capa.git /app/capa

COPY . /app/.

ENTRYPOINT ["bash", "/app/wrapper.bash"]
