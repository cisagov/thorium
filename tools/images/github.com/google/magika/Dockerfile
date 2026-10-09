ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir magika

COPY wrapper.bash tag.py /app/

ENTRYPOINT ["bash", "/app/wrapper.bash"]