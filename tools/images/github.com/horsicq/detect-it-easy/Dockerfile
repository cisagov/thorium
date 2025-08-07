ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR "/app"
COPY run_die.py /app/.

RUN apt update -qq && \
    apt install -y wget libglib2.0-0 libqt5sql5 libqt5svg5 libqt5gui5 libqt5widgets5 libqt5opengl5 libqt5dbus5 libqt5scripttools5 libqt5script5 libqt5network5 && \
    wget https://github.com/horsicq/DIE-engine/releases/download/3.10/die_3.10_Ubuntu_22.04_amd64.deb && \
    dpkg -i die_3.10_Ubuntu_22.04_amd64.deb && \
    rm die_3.10_Ubuntu_22.04_amd64.deb && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python3", "/app/run_die.py"]
