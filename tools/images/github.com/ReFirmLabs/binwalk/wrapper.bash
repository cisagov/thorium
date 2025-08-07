#!/bin/bash

mkdir -p /tmp/thorium/binwalk
binwalk -e $1 -C /tmp/thorium/binwalk > /tmp/thorium/results
mv /tmp/thorium/binwalk/*.extracted /tmp/thorium/children/carved/unknown/ || true

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
