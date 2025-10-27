#!/bin/bash

# run autovolatility3
./autovol3.py -f $1 -o /tmp/thorium/result-files -s full
# move our results to thorium
cd /tmp/thorium/result-files/volatility_analysis_* && mv * /tmp/thorium/result-files/.
# remove our volatiltity analysis folder
rm -rf /tmp/thorium/result-files/volatiltity_analysis_*
# list things in our result files dir for debug
ls /tmp/thorium/result-files
# remove our errors.txt file because it seems useless
rm /tmp/thorium/result-files/errors.txt
# move our info.txt file to our results file
mv /tmp/thorium/result-files/*_info.txt /tmp/thorium/results
