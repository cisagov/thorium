ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip3 install pefile

COPY dump_pefile.py /app/.

ENTRYPOINT ["python3", "/app/dump_pefile.py", "-o", "/tmp/thorium/results", "-t", "/tmp/thorium/tags"]