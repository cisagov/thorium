import sys
import json
import os
import subprocess
import re

def run(file_path):
    # get clamav version
    cmd = ['clamscan', '--version']
    clamav_version = subprocess.Popen(cmd,
                                      shell=False,
                                      stdout=subprocess.PIPE).communicate()[0]
    clamav_version = clamav_version.rstrip().decode('utf-8')

    # run clamav tool on sample
    cmd = ['clamscan', '--no-summary', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    err_code = proc.returncode

    clamav_output = re.search(r"{}".format("[^:]+: (.+) FOUND"), out.decode('utf-8'))

    if (clamav_output != None):
        clamav_output = clamav_output.group(1).lstrip().rstrip()

    # check for errors and upload tags
    tags = {}
    if err.decode('utf-8') != '':
        clamav_output = {'Errors': [f"AV Error({err_code}): {err.decode('utf-8')}"], 'Version': clamav_version}
    # upload tags for findings
    elif clamav_output != None:
        tags = {"ClamAV": clamav_output}
        clamav_output = {"Result": clamav_output, 'Version': clamav_version}
    else:
        clamav_output = {"Result": "Ok", 'Version': clamav_version}

    # return results dict
    return {'results': clamav_output, 'tags': tags}

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
