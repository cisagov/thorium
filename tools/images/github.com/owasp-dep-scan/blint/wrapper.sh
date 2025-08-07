#!/bin/sh

blint -i $1 -o /tmp/lint-results
blint sbom -i $1 -o /tmp/thorium/result-files/sbom.json

cp /tmp/lint-results/* /tmp/thorium/result-files/
cat /tmp/lint-results/blint-output.html > /tmp/thorium/results