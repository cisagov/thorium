#!/bin/bash

# trigger update of sources and rules
sed -z -i 's|unix-command:\n  enabled: auto|unix-command:\n  enabled: no|g' /etc/suricata/suricata.yaml
suricata-update update-sources
suricata-update --no-reload -f
