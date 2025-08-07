import argparse
import json
import pathlib

import capa.render.default
import capa.render.result_document


# this is a recursive function designed to extract layered children
# child argument is a dictionary
def extractChildrenMatches(child, matches):
    # only keep the current child if it has a location value
    currentChild = {}
    locations = child["locations"]
    if not locations == []:
        currentChild = child.copy()
        currentChild.pop("children")

        # cast all location values to hex
        for i in range(len(currentChild["locations"])):
            if "value" in currentChild["locations"][i]:
                try:
                    currentChild["locations"][i]["value"] = hex(currentChild["locations"][i]["value"])
                except:
                    pass
        matches.append(currentChild)

    # no subchildren means end recursion, add current child and return
    subChildren = child["children"]
    if subChildren == []:
        return matches

    # recursively get all matches for nested sub children
    for sc in subChildren:
        matches = extractChildrenMatches(sc, matches)

    return matches

# extracts the matches
def extractMatches(rule):
    hits = []
    matches = rule["matches"]
    for match in matches:
        for section in match:
            if "children" in section:
                topChild = section
                layeredSectionChildHits = extractChildrenMatches(topChild, [])
                hits.append(layeredSectionChildHits)
    return hits

def hasMbcDetection(rule):
    return not rule["meta"]["mbc"] == []

def hasAttackDetection(rule):
    return not rule["meta"]["attack"] == []

# extract rule tag from raw mbc rules
def filterMbc(rawRule):
    data = rawRule["meta"]["mbc"][0] #raw rules come in a list
    objective = data["objective"]
    behavior = data["behavior"]
    IDTag = data["id"]
    ruleString = f"{objective}::{behavior} {IDTag}"
    return ruleString

# extract rule tag from raw attack rules
def filterAttack(rawRule):
    data = rawRule["meta"]["attack"][0] # raw rules come in a list
    tactic = data["tactic"]
    technique = data["technique"]
    IDTag = data["id"]
    ruleString = f"{tactic}::{technique} {IDTag}"
    return ruleString

# performs a crude scan of all fields to see if there are any valid attack/mbc tags
# splits rules accordingly
def extractRaw(capaResults):
    rawAttackRules = []
    rawMbcRules = []

    if "rules" in capaResults:
        rules = capaResults["rules"]
        for key in rules:
            rule = rules[key]
            if hasAttackDetection(rule):
                rawAttackRules.append(rule)
            if hasMbcDetection(rule):
                rawMbcRules.append(rule)

    return rawAttackRules, rawMbcRules

def postprocess(raw_results):
    with open(raw_results, "r") as f:
        capaResults = json.load(f)

    attackMatches = []
    mbcMatches = []
    attackTags = []
    mbcTags = []

    # get tags and matches from raw capa json results
    rawAttack, rawMbc = extractRaw(capaResults)
    for r in rawAttack:
        attackTag = filterAttack(r)
        attackTags.append(attackTag)
        attackMatches.append({attackTag : extractMatches(r)})

    for r in rawMbc:
        mbcTag = filterMbc(r)
        mbcTags.append(mbcTag)
        mbcMatches.append({mbcTag : extractMatches(r)})

    # combine results
    results = {}
    tags = {"ATT&CK": attackTags, "MBC": mbcTags}
    results["results"] = {"ATT&CK" : attackMatches ,"MBC" : mbcMatches}
    results["tags"] = tags

    return results


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Postprocess raw CAPA results.")

    # Add the raw results argument
    parser.add_argument("raw_results", type=str, help="Input for raw results")

    # Add flags for paths
    parser.add_argument("--tags", type=str, default="/tmp/thorium/tags",
                        help="Path to the tags directory (default: /tmp/thorium/tags)")
    parser.add_argument("--results", type=str, default="/tmp/thorium/result-files/results.json",
                        help="Path to the results JSON file (default: /tmp/thorium/result-files/results.json)")

    # Parse the arguments
    args = parser.parse_args()

    # post-process to extract our desired tags
    output = postprocess(args.raw_results)

    # write abbreviated results to thorium results path
    with open(args.results, "w") as f:
        json.dump(output["results"], f)

    if output.get("tags"):
        # write tool results to thorium results path
        with open(args.tags, "w") as f:
            json.dump(output["tags"], f)

    # output the default capa text for the result document
    doc = capa.render.result_document.ResultDocument.from_file(pathlib.Path(args.raw_results))
    capa.render.default.render_default(doc)

if __name__ == "__main__":
    main()
