ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

RUN pip3 install --no-cache-dir xortool importlib_metadata

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]