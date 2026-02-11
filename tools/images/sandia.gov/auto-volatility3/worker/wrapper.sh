#!/bin/bash

echo "vol -vvv -q --remote-isf-url 'https://github.com/Abyss-W4tcher/volatility3-symbols/raw/master/banners/banners.json' -f $3 $1 > /tmp/thorium/cache/files/$2"
vol -vvv -q --remote-isf-url 'https://github.com/Abyss-W4tcher/volatility3-symbols/raw/master/banners/banners.json' -f $3 $1 > /tmp/thorium/cache/files/$2
