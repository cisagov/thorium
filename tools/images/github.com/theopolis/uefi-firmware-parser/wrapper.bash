#! /bin/bash

python3 /app/uefi-firmware-parser.py --extract --output /tmp/thorium/children/unpacked/ --structure /tmp/thorium/result-files/structure.json $1 > /tmp/thorium/results