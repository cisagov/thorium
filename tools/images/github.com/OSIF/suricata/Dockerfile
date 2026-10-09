FROM jasonish/suricata:latest as base
FROM docker.io/almalinux/amd64:9-base

# copy over pre-built layers to remove volumes that make some files ephemeral
COPY --from=base / /

WORKDIR /suricata

COPY update.bash /suricata/update.bash
COPY run_suricata.bash /suricata/run_suricata.bash
COPY summarize.py /suricata/summarize.py

RUN bash /suricata/update.bash

ENTRYPOINT ["bash", "/suricata/run_suricata.bash"]