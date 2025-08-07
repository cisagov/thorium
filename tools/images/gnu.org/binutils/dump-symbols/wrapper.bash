#! /bin/sh

set -e

/usr/bin/nm --quiet --defined-only --demangle $@ > /tmp/thorium/result-files/symbols.txt

if [ ! -s /tmp/thorium/result-files/symbols.txt ]; then
    echo "No symbols found" | tee /tmp/thorium/results

    echo '{"HasSymbols": "False"}' > /tmp/thorium/tags

    rm /tmp/thorium/result-files/symbols.txt
    exit 0
fi

line_count=$(wc -l /tmp/thorium/result-files/symbols.txt)
echo "Found ${line_count} symbols" | tee /tmp/thorium/results

echo '{"HasSymbols": "True"}' > /tmp/thorium/tags
