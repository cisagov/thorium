#!/bin/bash

RESULTS=$(magika -s $1)
RESULT_FILES=$(magika --json $1 | awk '{$1=""}1' | awk '{$1=$1};1')

if [ $? -ne 0 ]; then
    echo $RESULTS
    RESULTS=$(magika-python-client -s $1 | awk '{$1=""}1' | awk '{$1=$1};1')
    RESULT_FILES=$(magika-python-client --json $1) 
fi
echo $RESULT_FILES > /tmp/thorium/result-files/magika.json
echo $RESULTS > /tmp/thorium/results
