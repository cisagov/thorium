ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR /app

RUN pip3 install blint --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY wrapper.sh /app/.

ENTRYPOINT ["/bin/sh", "/app/wrapper.sh"]