ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

RUN apt update && \
    apt install -y python3 python3-pip automake libtool make gcc pkg-config flex bison git openssl libssl3 libssl-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://github.com/VirusTotal/yara/archive/refs/tags/v4.5.5.tar.gz /app/

RUN tar -xzvf v4.5.5.tar.gz && \
    rm v4.5.5.tar.gz

WORKDIR /app/yara-4.5.5

RUN ./bootstrap.sh && \
    ./configure && \
    make && \
    make install

# solve cannot open shared object file error
RUN cp /usr/local/lib/libyara* /usr/lib/

# install yara python library
#RUN python3 -m pip install -U pip && \
#    python3 -m pip install yara

# copy yara rules to the image
COPY . /app/.

WORKDIR "/app/rules"
RUN mkdir -p /app/rules/allRules
RUN git clone https://github.com/kevoreilly/CAPEv2
RUN git clone https://github.com/Yara-Rules/rules
RUN git clone https://github.com/Neo23x0/signature-base
# No longer maintained rule repos but could still be useful
RUN git clone https://github.com/reversinglabs/reversinglabs-yara-rules
RUN git clone https://github.com/CyberDefenses/CDI_yara
RUN git clone https://github.com/nccgroup/Cyber-Defence

RUN find /app/rules -name "*.yar*" ! -path "/app/rules/allRules/*" -exec cp "{}" /app/rules/allRules \;

# compile yara rules
RUN python3 renameRules.py allRules
RUN ./compile.sh /app/rules.bin
# clean up rules projects that were cloned
RUN rm -rf /app/rules

WORKDIR "/app"

ENTRYPOINT ["python3", "/app/run_yara.py"]