#!/bin/bash

# Check if encryption is found
if 7z l -slt $1 | grep -q ZipCrypto; then
  echo "ZIP is encrypted, checking passwords against dictionary..."

  PASSWORD=$(fcrackzip -D -p dictionary.txt -u $1 | awk '{print $5}' | xargs)
  echo "Using Password: $PASSWORD"
  unzip -o -d /tmp/thorium/children/unpacked -P "$PASSWORD" $1

  echo '{"Encrypted": "True"}' > /tmp/thorium/tags
# No encryption is found, don't pass in password to unzip
else
  echo "No ZIP encryption found, attempting to unzip..."
  unzip -o -d /tmp/thorium/children/unpacked  $1
fi

echo "Unzipped the following files: \n" > /tmp/thorium/results
echo $(ls /tmp/thorium/children/unpacked) >> /tmp/thorium/results
