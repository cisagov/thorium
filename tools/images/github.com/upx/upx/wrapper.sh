#!/bin/sh

FNAME=$(basename $1)

echo "Unpacking $FNAME"

upx -o /tmp/thorium/children/unpacked/$FNAME -d $1