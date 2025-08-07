#!/bin/sh

echo Running surfactant on $1

RESULTS_PATH="/tmp/thorium/result-files"

surfactant generate $1 $RESULTS_PATH/output.sbom
# make a copy into the results file
mv $RESULTS_PATH/output.sbom /tmp/thorium/results