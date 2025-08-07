import argparse
import json
import pefile

# Recursively check each key and value of a dictionary to ensure it is json serializable
def decode_dict(raw):
  fixed = dict()
  
  # decode all keys and values of the dictionary
  if isinstance(raw, dict):
    for key, value in raw.items():
      # decode each key within the dictionary
      if isinstance(key, (bytes, bytearray)):
        key = key.decode('utf-8')
      # recursively call decode_dict
      if isinstance(value, dict):
        value = decode_dict(value)
      elif isinstance(value, (list, tuple, set)):
        value = [decode_dict(member) for member in value]
      fixed[key] = decode_dict(value)
    return fixed
  
  # call decode_bytes on each member of lists, tuples or sets
  if isinstance(raw, (list, tuple, set)):
    return [decode_dict(member) for member in raw]
  # decode byte or bytearray values
  elif isinstance(raw, (bytes, bytearray)):
    return raw.decode('utf-8')
  # return all other types
  return raw


def main(infile: str, outfile: str = None, tags: str = None) -> None:
  # load input file into pefile library
  pe = pefile.PE(infile)
  
 
  # write imphash as a json formated key/value tag pair
  if (tags):
    with open(tags, 'w') as f:
      # create imphash thorium tag
      imphash_tag = {'Imphash': f'{pe.get_imphash()}'}
      json.dump(imphash_tag, f)

  # decode any byte or bytearray values in the dictionary dump from pefile
  json_dict = decode_dict(pe.dump_dict())
  
  # put parsing warnings into a key Thorium can see
  if "Parsing Warnings" in json_dict:
    json_dict['Warnings'] = json_dict.pop("Parsing Warnings")

  # dump fixed json to stdout or file depending on kargs
  if (outfile):
    # write json formatted pefile dump 
    with open(outfile, 'w') as f:
      json.dump(json_dict, f)
  else:
    print(json.dumps(json_dict))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--outfile', help='The file to write the results to', type=str, required=False)
  parser.add_argument('-t', '--tags', help='The file to write the key/value tags to', type=str, required=False)
  parser.add_argument('file', help='The file to analyze with pefile', type=str)
  args = parser.parse_args()
  main(args.file, args.outfile, args.tags)
