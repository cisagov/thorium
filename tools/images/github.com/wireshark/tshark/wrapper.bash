#!/bin/bash

CHILDREN_PATH="/tmp/thorium/children/carved/pcap"
mkdir -p $CHILDREN_PATH
/usr/bin/capinfos "$1" > /tmp/thorium/results

# extract files from supported protocols
protos="http tftp imf smb dicom"
for PROTO in $protos; do
    mkdir -p $CHILDREN_PATH/$PROTO
    tshark -r "$1" --export-object "$PROTO,$CHILDREN_PATH/$PROTO"
done
