#!/bin/sh

# copy our trivy cache to our cache folder if it hasn't already been moved
[ -d "/tmp/cache" ] && mv /tmp/cache/* /root/.cache/ && rm -rf /tmp/cache
# run trivy and dump our table results
trivy fs --scanners vuln,secret,misconfig,license --include-dev-deps -f table -o /tmp/thorium/results $1
# run trivy and dump our json results
trivy fs --scanners vuln,secret,misconfig,license --include-dev-deps -f json -o /tmp/thorium/result-files/trivy.json $1