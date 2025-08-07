ARG IMAGE="ubuntu:22.04"
FROM $IMAGE AS clone 

WORKDIR "/tmp"

RUN apt update && apt install -y unzip

ADD https://github.com/mandiant/flare-floss/releases/download/quantumstrand-beta1/quantumstrand-quantumstrand-beta1-linux.zip /tmp/

RUN unzip /tmp/quantumstrand-quantumstrand-beta1-linux.zip

FROM $IMAGE

WORKDIR "/app"

COPY --from=clone /tmp/quantumstrand /app/quantumstrand
COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]