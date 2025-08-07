import sys
import json
import os
import subprocess

def run(file_path):
    # run capa tool on sample
    cmd = ['/app/capa', '-q', file_path]
    proc = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    errcode = proc.returncode
    decout = out.decode("utf-8", errors="replace")
        
    # process Capa results
    if decout.strip() == "":
        decerr = err.decode("utf-8", errors="replace")
        warning = f"Warning: Capa could not process {file_path}"
        decout = f"{warning}:\n{decerr}"

    # cleanup output file from capa
    if os.path.exists(file_path + ".viv"):
        os.remove(file_path + ".viv")

    tags = {}
    # upload tags for capa output
    if "This sample appears to be packed" in decout:
        tags = {"Packed": f"True"}

    # return results dict
    return {'results': decout, 'tags': tags}

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
