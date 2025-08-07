#!/bin/bash

mkdir -p /tmp/thorium/carved/
mkdir -p /tmp/thorium/children/carved/unknown
foremost -V -i $1 -o /tmp/thorium/carved
mv /tmp/thorium/children/carved/audit.txt /tmp/thorium/results
mv /tmp/thorium/children/carved/* /tmp/thorium/children/carved/unknown/ || true

binHash=$(sha256sum $1)
binHashArr=($binHash)
for f in /tmp/thorium/children/carved/unknown/*/*; do
    fileHash=$(sha256sum $f)
    fileHashArr=($fileHash)
    if [[ "${fileHashArr[0]}" == "${binHashArr[0]}" ]]
        then
                echo "Removing Duplicate Sample" $f
                rm $f
        fi
done
