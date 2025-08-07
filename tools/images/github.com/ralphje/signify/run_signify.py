from signify import exceptions as sig_except
import json
import pe_certs
import sys

def run(file_path):
    output = {}
    try:
        # use signify library to extract signature info from sample
        output = pe_certs.get_certs_from_file(file_path)
    # catch invalid signer version
    except sig_except.SignerInfoParseError as e:
        output = {"Errors": [f"{e}"]}
    # signify seems to have a bug with trying to get the group attribute on a NoneType
    except AttributeError as e:
        output = {"Errors": [f"{e}"]}
    except Exception as e: 
        output = {"Errors": [f"{e}"]}
    return output

if len(sys.argv) == 1:
    output = {"Errors": [f"No sample path provided"]}
else:
    output = run(sys.argv[1])

with open("/tmp/thorium/results", "w") as f:
    f.write(json.dumps(output))
