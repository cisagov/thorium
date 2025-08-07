import sys
import json
import os
import subprocess

# combines two tag dictionaries into one
# assumes that values are arrays
def combineTags(tags1, tags2):
    for key in tags2:
        if not key in tags1:
            tags1[key] = tags2[key]
        else:
            tags1[key].append(tags2[key][0])

    return tags1

def extractTags(detection):
    tags = {}
    if "extension" in detection:
        tags["FileTypeExtension"] = [detection["extension"]]
    if "value" in detection:
        tags["FileTypeMatch"] = [detection["value"]]
    if "name" in detection:
        tags["MIMEType"] = [detection["name"]]
        tags["FileType"] = []
        for subElement in detection['subEls']:
            if ("type" in subElement):
                tags["FileType"].append(subElement["type"].upper())
        if len(tags["FileType"]) == 0:
            del tags["FileType"]

    tags = trimTags(tags)
    return tags

# strip whitespace from tags
def trimTags(tags):
    trimmedTags = {}
    for key in tags:
        tagList = tags[key]
        newList = []
        for i in tagList:
            newList.append(i.strip())
        trimmedTags[key] = newList
    return trimmedTags

# do final processing/filtering on tags
def doProcessing(tags):
    finalTags = {}
    
    for key in tags:
        value = tags[key]
        if len(value) < 2:
            finalTags[key] = value[0]
        else:
            finalTags[key] = value

    return finalTags

def getTags(resultJson):
    tags = {}
    counter = 0

    if "struc" in resultJson:
        detectionData = resultJson['struc']
        for detection in detectionData:
            extractedTags = extractTags(detection)
            tags = combineTags(tags, extractedTags)
            counter += 1

    if counter > 1:
        tags['Polyglot'] = True

    return tags

# polyfile INPUT_FILE --format sbud --output output.json
def run(filepath):
    fileName = os.path.basename(filepath)
    jsonOutput = f"/tmp/thorium/result-files/polyfile-{fileName}.json"
    htmlOutput = f"/tmp/thorium/result-files/hexviewer-{fileName}.html"
    resultOutput = "/tmp/thorium/results"

    result = {}
    cmd = [
        'polyfile',
        '--format', 'html',
        '--output', htmlOutput,
        '--format', 'explain',
        '--output', resultOutput,
        '--format', 'json',
        '--output', jsonOutput,
        filepath]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    errcode = proc.returncode

    # read in JSON to build tags
    rawData = open(jsonOutput, "r").read()

    with open(jsonOutput, "w") as f:
        resultJson = json.loads(rawData)
        resultJson.pop('b64contents', None)
        for element in resultJson['struc']:
            for subEl in element['subEls']:
                subEl.pop('value', None)
        f.write(json.dumps(resultJson))
        # tagging can be enabled here, but lots of false positives
        # tags = getTags(resultJson)
        # result['tags'] = tags

    return result

# the path to the sample is taken as an input
if len(sys.argv) == 1:
    output = {'results': {"Errors": [f"No sample path provided"]}}
else:
    output = run(sys.argv[1])

if 'tags' in output and output['tags'] != {}:
    # write tool results to thorium results path
    with open("/tmp/thorium/tags", "w") as f:
        f.write(json.dumps(output['tags']))

