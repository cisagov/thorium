#!/bin/sh

mimetype -b $1 > /tmp/thorium/results
cat /tmp/thorium/results | xargs -I## echo '{"MIMEType": ["##"]}' > /tmp/thorium/tags 