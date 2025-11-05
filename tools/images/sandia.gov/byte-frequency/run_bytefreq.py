import sys
import json
import bytefreq_utils as utils

def run(file_path):

    # run bytefreq tool on sample and create graph file
    results = dict()
    results["bytecount"] = utils.calcByteFreq(file_path)
    results["variance"] = utils.calcVariance(results["bytecount"])
    file_name = file_path.split('/')[-1]
    graph_name = f'freq-graph-{file_name}.png'
    graph_path = f'/tmp/thorium/result-files/{graph_name}'
    utils.makeByteFreqPlot(results["bytecount"], graph_path)
    return results

# the path to the sample is taken as an input
if len(sys.argv) == 1:
    output = {"Errors": [f"No sample path provided"]}
else:
    output = run(sys.argv[1])

# write tool results to thorium results path
with open("/tmp/thorium/results", "w") as f:
    f.write(json.dumps(output))
