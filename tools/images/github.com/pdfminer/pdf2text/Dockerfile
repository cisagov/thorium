ARG IMAGE="python:3.11-slim"
FROM $IMAGE

RUN pip install --no-cache-dir pdfminer.six 'pdfminer.six[image]'

ENTRYPOINT ["pdf2txt.py", "-o", "/tmp/thorium/results"]