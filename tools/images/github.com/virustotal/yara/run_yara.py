import sys
import json
import os
import subprocess
import re

def yara_formatter(raw: str):
    '''
    formats the results from yara

    Args:
        raw (str): the raw string to format
    Returns:
        list: formatted list of hits
    '''
    regex = re.compile(r'([^ ]+) \[(.*)\] (\[.*\]) .*')
    hits = []
    lines = raw.splitlines()
    for l in lines:
        entry = {}
        m = regex.search(l)
        if m:
            entry['rule'] = m.group(1)
            entry['tags'] = m.group(2).split(',')
            entry['meta'] = m.group(3)
            hits.append(entry)
    return hits

def run(file_path):
    # run yara tool on sample
    cmd = ['yara', '-w', '-g', '-m', '-C', 'rules.bin', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    proc.wait()
    out, err = proc.communicate()
    errcode = proc.returncode

    # return results
    # decode and format results
    decout = out.decode("utf-8")
    formatted = yara_formatter(decout)

    tags = {}
    rules = []
    # gather yara rule hits
    for hit in formatted:
        rules.append(hit["rule"])
    # if yara had hits, then submit rule string as a tag
    if rules:
        tags = {"YaraRuleHits": rules}

    # set result to none if no hits found
    if formatted == []:
        formatted = {'result': 'no hits'}

    # return results dict
    return {'results': formatted, 'tags': tags}

# the path to the sample is taken as an input
if len(sys.argv) == 1:
    output = {'results': {"Errors": [f"No sample path provided"]}}
else:
    output = run(sys.argv[1])

# write tool results to thorium results path
with open("/tmp/thorium/results", "w") as f:
    f.write(json.dumps(output['results']))

if 'tags' in output and output['tags'] != {}:
    # write tool results to thorium results path
    with open("/tmp/thorium/tags", "w") as f:
        f.write(json.dumps(output['tags']))
