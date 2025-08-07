FROM jasonish/suricata:latest as base
FROM docker.io/almalinux/amd64:9-base

# copy over pre-built layers to remove volumes that make some files ephemeral
COPY --from=base / /

WORKDIR /suricata

ADD update.bash /suricata/update.bash
ADD run_suricata.bash /suricata/run_suricata.bash
ADD summarize.py /suricata/summarize.py

RUN bash /suricata/update.bash

ENTRYPOINT ["bash", "/suricata/run_suricata.bash"]