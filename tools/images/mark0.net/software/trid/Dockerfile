ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

RUN apt update && \
    apt install -y unzip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# get the executable
ADD https://mark0.net/download/trid_linux_64.zip /app
# get the database
ADD https://mark0.net/download/triddefs.zip /app
RUN unzip triddefs.zip && \
    rm triddefs.zip && \
    unzip trid_linux_64.zip && \
    rm trid_linux_64.zip && \
    chmod a+x trid

# set env variables so trid can run and not crash
ENV LC_ALL=C

COPY run_trid.py /app/run_trid.py

ENTRYPOINT ["python3", "run_trid.py"]
