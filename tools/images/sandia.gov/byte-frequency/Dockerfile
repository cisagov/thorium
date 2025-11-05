ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

COPY . /app/.

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "/app/run_bytefreq.py"]