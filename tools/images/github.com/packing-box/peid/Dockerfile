ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

RUN pip3 install peid

COPY run_peid.py /app/.

ENTRYPOINT ["python3", "/app/run_peid.py"]