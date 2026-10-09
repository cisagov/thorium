#!/bin/bash

# foremost refuses a non-empty output dir, so carve first and sort into unknown/ after
mkdir -p /tmp/thorium/children
foremost -v -i "$1" -o /tmp/thorium/children/carved
mv /tmp/thorium/children/carved/audit.txt /tmp/thorium/results
mkdir -p /tmp/thorium/children/carved/unknown
find /tmp/thorium/children/carved -mindepth 1 -maxdepth 1 ! -name unknown -exec mv -t /tmp/thorium/children/carved/unknown/ {} +

binHash=$(sha256sum "$1")
binHashArr=($binHash)
for f in /tmp/thorium/children/carved/unknown/*/*; do
    [ -f "$f" ] || continue
    fileHash=$(sha256sum "$f")
    fileHashArr=($fileHash)
    if [[ "${fileHashArr[0]}" == "${binHashArr[0]}" ]]
        then
                echo "Removing Duplicate Sample" $f
                rm "$f"
        fi
done
