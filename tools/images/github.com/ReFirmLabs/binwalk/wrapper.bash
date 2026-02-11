#!/bin/bash

mkdir -p /tmp/thorium/binwalk
binwalk -e $1 -C /tmp/thorium/binwalk > /tmp/thorium/results

if find "/tmp/results/binwalk" -maxdepth 1 -type f -name "*.extracted" -print -quit | grep -q .; then
    mv /tmp/thorium/binwalk/*.extracted /tmp/thorium/children/carved/unknown/ 
else
    echo "INFO: No files extracted"
    exit 0
fi

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
