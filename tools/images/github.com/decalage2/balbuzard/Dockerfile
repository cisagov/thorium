ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

RUN apt update && \
    apt install python2 git -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


ADD https://bootstrap.pypa.io/pip/2.7/get-pip.py get-pip.py
RUN python2 get-pip.py
RUN pip2 install balbuzard
RUN git clone https://github.com/decalage2/balbuzard.git

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]