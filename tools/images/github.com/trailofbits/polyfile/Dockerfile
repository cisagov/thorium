ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip3 install --no-cache-dir polyfile

COPY run_polyfile.py /app/.

ENTRYPOINT ["python3", "/app/run_polyfile.py"]