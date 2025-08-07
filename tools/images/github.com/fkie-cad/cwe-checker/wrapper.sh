#!/bin/sh

cp -R /home/cwe/.config ~/
cp -r /home/cwe/.local ~/
/home/cwe/cwe_checker -j -v -o /tmp/thorium/results $1