#!/bin/bash

# dump .dbinfo and format as JSON
sqlite3 $1 ".dbinfo" | sed 's/  */ /g' | sed 's/data version/data version:/g' | jq -Rn '
  [inputs | split(": ") | { (.[0]): .[1] }] | add
' > /tmp/thorium/results

# dump schema to result file
sqlite3 $1 .schema >> /tmp/thorium/result-files/schema.sql

# dump non-system table names and add to .dbinfo json dictionary
TABLES=$(sqlite3 $1 .tables | awk '{$1=$1};1' | tr -d "\n")
META_JSON=$(jq --arg newArray "$TABLES" '. + {tables: ($newArray | split(" "))}' /tmp/thorium/results)
echo $META_JSON > /tmp/thorium/results