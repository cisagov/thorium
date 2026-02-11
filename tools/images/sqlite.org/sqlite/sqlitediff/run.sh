#!/bin/bash

sqldiff $1 $2 > /tmp/thorium/result-files/diff.sql
sqldiff $2 $1 > /tmp/thorium/result-files/reverse-diff.sql
sqldiff --summary $1 $2 > /tmp/thorium/results