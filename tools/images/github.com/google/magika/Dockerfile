ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip install --upgrade pip && \
    pip install magika && \
    rm -rf /root/.cache/pip

COPY wrapper.bash /app/.

ENTRYPOINT ["bash", "/app/wrapper.bash"]