#!/bin/sh

cp $1 $1.pyc
uncompyle6 -o /tmp/thorium/ $1.pyc
ls -la /tmp/thorium/results