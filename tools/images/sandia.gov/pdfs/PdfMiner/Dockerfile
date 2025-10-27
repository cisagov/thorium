ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

RUN python3 -m pip install pdfminer.six

COPY pdfmine.py /app/.

ENTRYPOINT ["python3", "/app/pdfmine.py"]