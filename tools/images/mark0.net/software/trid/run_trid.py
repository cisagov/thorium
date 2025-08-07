import sys
import json
import subprocess

# tool will only tag detections above this limit (25%)
tagDetectionLimit = 25


def aboveDetectionThreshold(line):
    split = line.split("%")
    detectionValue = int(float(split[0]))
    return detectionValue > tagDetectionLimit

# extracts tags of matches by newlines
def extractTags(results):
    tags = {}
    tags['FileTypeMatch'] = []
    split = results.split("\n")
    for s in split:
        if "%" in s and aboveDetectionThreshold(s):
            lineSplit = s.split(" ")
            # cut off percentage part of results and last chunk
            nameSection = lineSplit[3:len(lineSplit) - 1]
            name = " ".join(nameSection)
            tags['FileTypeMatch'].append(name)
    return tags


def run(file_path):
    cmd = ['./trid', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    errcode = proc.returncode
    # process results
    results = out.decode("utf-8")

    tags = extractTags(results)

    # return results dict
    return {'results': results, 'tags': tags}

# the path to the sample is taken as an input
if len(sys.argv) == 1:
    output = {'results': {"Errors": [f"No sample path provided"]}}
else:
    output = run(sys.argv[1])

# write tool results to thorium results path
with open("/tmp/thorium/results", "w") as f:
    f.write(output['results'])

if 'tags' in output and output['tags'] != {}:
    # write tool results to thorium results path
    with open("/tmp/thorium/tags", "w") as f:
        f.write(json.dumps(output['tags']))
