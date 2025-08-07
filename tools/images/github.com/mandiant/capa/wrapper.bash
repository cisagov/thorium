#! /bin/bash

set -e

capa -s /app/capa/sigs -r /app/capa-rules -j $1 > /tmp/thorium/result-files/raw_results.json

python3 postprocess.py --results /tmp/thorium/results /tmp/thorium/result-files/raw_results.json

# needed?
if [ -f "$1.viv" ]; then
    rm "$1.viv"
fi