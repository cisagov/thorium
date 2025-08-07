#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

out=$1

if [ -z $out ];then
    echo "Usage $0 <output binary rule file>"
    exit -1
fi

# remove bad rules
for f in allRules/* ; do
  yarac $f a
  if [ $? != 0 ]
  then
    echo "Bad Rule: $f"
    rm $f
  fi
done

# removing false positive rules, temp fix, need to do something more permanent later
rm allRules/ip.yar || True
rm allRules/domain.yar || True
rm allRules/base64.yar || True


echo "Compiling Valid Rules Now..."

# compile all rules
yarac $DIR/allRules/*.yar* $out
