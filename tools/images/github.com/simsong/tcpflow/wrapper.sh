#!/bin/sh

mkdir -p /tmp/thorium/children/carved/pcap
tcpflow -o /tmp/thorium/children/carved/pcap -r $1
mv /tmp/thorium/children/carved/pcap/report.xml /tmp/thorium/results