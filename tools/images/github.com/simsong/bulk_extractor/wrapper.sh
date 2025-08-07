#/bin/sh

mkdir -p /tmp/thorium/bulkout
bulk_extractor $1 -o /tmp/thorium/bulkout
mv /tmp/thorium/bulkout/report.xml /tmp/thorium/results
mv /tmp/thorium/bulkout/*.txt /tmp/thorium/result-files/ || true
mv /tmp/thorium/bulkout/* /tmp/thorium/children/carved/unknown/ || true
rm /tmp/thorium/children/carved/unknown/$(basename $1) || true