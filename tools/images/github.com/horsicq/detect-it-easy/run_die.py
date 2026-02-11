import sys
import json
import subprocess


# pulls the top level tags from the detection results
def pullDetectionTags(detections):
    tags = {}
    
    # casting tag to camel case
    if detections:
        if 'filetype' in detections:
            tags['FileType'] = detections['filetype']

    return tags

# pulls the tags inside the values
def pullValueTags(values):
    tags = {}

    for v in values:
        if 'type' in v and 'string' in v:
            valueType = v['type']
            valueString = v['string']
            valueString = valueString.replace(f"{valueType}: ", "")
            # Cast some keys
            if valueType == 'Sign tool':
                valueType = 'SignTool'
            elif valueType == 'Format':
                valueType = 'FileFormat'
            tags[valueType] = valueString

    return tags

def combineTags(tagList1, tagList2):
    result = {}

    # add tag list 1
    for key in tagList1:
        if not key in result:
            result[key] = [tagList1[key]]
        else:
            result[key].append(tagList1[key])

    # add tag list 2
    for key in tagList2:
        if not key in result:
            result[key] = [tagList2[key]]
        else:
            result[key].append(tagList2[key])
        result[key] = tagList2[key]

    return result

def run(file_path):
    # run DetectItEasy tool on sample 
    # | grep -v "TypeError:"
    cmd = ['diec', '-j', file_path]
    process = subprocess.run(cmd,
                             text=True,
                             capture_output=True,
                             check=True)
    # process DetectItEasy results
    resultLines = process.stdout.split("\n")
    results = ""
    for line in resultLines:
        if (not line.startswith("[!]")):
            results += line
    results_json = json.loads(results)

    # detections actually returns a list, thus the need for [0]. The actual value is a json
    detections = {}
    # make sure detects exists and there was a valid value present
    if 'detects' in results_json and isinstance(results_json['detects'], list) and len(results_json['detects']) > 0:
        detections = results_json['detects'][0]
    detectionTags = pullDetectionTags(detections)

    # values are located in the 'detects' json one level down
    # all detection really holds is the filetype

    if 'values' in detections:
        values = detections['values']
        valueTags = pullValueTags(values)
        detectionTags = combineTags(detectionTags, valueTags)

    # return results dict
    return {'results': results_json, 'tags': detectionTags}

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
