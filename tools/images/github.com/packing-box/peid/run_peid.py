import sys
import json
import subprocess

out_filepath = "/tmp/thorium/results"

# filter a list to return valid lines
def filterValidLines(split):
    validLines = []
    for s in split:
        if len(s) > 1:
            validLines.append(s)

    return validLines

# extract tags from die output
def extractTags(strOutput):
    tags = {}

    split = strOutput.split("\n")
    uniques = list(dict.fromkeys(split))

    # filter results
    validTags = filterValidLines(uniques)
    tags['Detections'] = validTags

    return tags

def extractError(errOutput):
    split = errOutput.split("\n")
    validLines = filterValidLines(split)
    size = len(validLines)

    # return the last line where the error is
    return split[size - 1]

def run(file_path):
    results = ""
    tags = {}

    cmd = ['peid', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    errcode = proc.returncode

    # process peid results
    strOutput = out.decode('utf-8')
    errOutput = err.decode('utf-8')
    if errcode == 0:
        results = strOutput
        tags = extractTags(strOutput)
    elif "DOS Header magic not found" in errOutput:
        print(errOutput)
        results = extractError(errOutput)
    else:
        print(errOutput)
        exit(int(errcode)) # fail the tool

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
