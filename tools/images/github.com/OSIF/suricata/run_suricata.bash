#!/bin/bash

PCAP=$1
mkdir -p /tmp/thorium/result-files
cd /tmp/thorium/result-files
suricata -c /etc/suricata/suricata.yaml -r $1
cat /tmp/thorium/result-files/suricata.log
python3 /suricata/summarize.py
cat /tmp/thorium/result-files/stats.log | sed 's/|/,/g' | sed 's/ //g' | sed 's/Date:/##### Date: /g' | grep -v -E "[-]{3}" >> /tmp/thorium/results