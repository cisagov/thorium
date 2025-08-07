#!/bin/sh

# needed to get pip install of unblob as unblob user to work
export HOME="/home/unblob"
/usr/local/bin/unblob --report /tmp/thorium/results -e /tmp/thorium/children/unpacked/ --log /tmp/thorium/result-files/unblob.log $1