#!/bin/bash

./trufflehog --no-update git file://$1 --json > /tmp/thorium/results