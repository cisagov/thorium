ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

RUN pip3 install uncompyle6

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]