ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip install --upgrade pip && \
    python3 -m pip install signify pyasn1

COPY . /app/.

ENTRYPOINT ["python3", "/app/run_signify.py"]
