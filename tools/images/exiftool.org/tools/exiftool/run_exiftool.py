import sys
import json
import os
import subprocess

# extract tags from results if they exist
def extractTags(result):
    tags = {}
    keys = ["FileType", "FileTypeExtension", "FileSize", "PEType", "MachineType", "EntryPoint", "MIMEType"]

    for k in keys:
        if (result and k in result):
            if len(result[k]) > 0:
                tags[k] = result[k]

    return tags

def run(file_path):
    # run exiftool tool on sample
    cmd = ['./exiftool/exiftool', '-j', '-b', '-api', 'timezone=UTC', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    errcode = proc.returncode
    # process ExifTool results
    decout = out.decode("utf-8", errors="replace")

    # get first element returned by ExifTool
    result = json.loads(decout)[0]

    # cleanup keys we don't want to appear in results
    exclude_keys = \
        ["Directory",
        "FileName",
        "FilePermissions",
        "SourceFile",
        "FileModifyDate",
        "FileAccessDate"
        "FileInodeChangeDate"]
    for result_key in exclude_keys:
        result.pop(result_key, None) 

    tags = extractTags(result)

    # return results dict
    return {'results': result, 'tags': tags}

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
