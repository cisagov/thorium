ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN pip3 install polyfile && \
    rm -rf /root/.cache/pip

COPY run_polyfile.py /app/.

ENTRYPOINT ["python3", "/app/run_polyfile.py"]