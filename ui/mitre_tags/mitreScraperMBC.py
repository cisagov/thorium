#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests


def cutByPhrases(text, phrase1, phrase2):
    cutResults = []
    remainingText = text
    while text.__contains__(phrase1) and text.__contains__(phrase2):
        index1 = text.find(phrase1)
        index2 = text.find(phrase2)
        cut = text[index1 + len(phrase1):index2]
        cutResults.append(cut)
        text = text[index2 + len(phrase2):]

    return cutResults

def extractPattern(s, phrase1, phrase2):
    return s[s.find(phrase1) + len(phrase1):s.find(phrase2)]

def extractByKeyword(inputList, keyword):
    extractedRows = []
    for line in inputList:
        if line.__contains__(keyword):
            extractedRows.append(line)
    return extractedRows

def extractTechnique(chunk):
    split = chunk.split("\n")
    dataLines = extractByKeyword(split, "<tspan")

    # clean data
    for i in range(len(dataLines)):
        dataLines[i] = extractPattern(dataLines[i], ">", "</tspan")

    result = dataLines[0]
    for line in dataLines[1:]:
        result += f" {line}" # add a space to this because formatting

    techniqueSplit = result.split(":")
    return techniqueSplit[0].strip(), techniqueSplit[1].strip()

def extractTactic(chunk):
    split = chunk.split("\n")
    for s in split:
        if s.__contains__("tspan"):
            return extractPattern(s, ">", "</tspan")

def postProcessing(allResults):
    for tactic in allResults:
        group = allResults[tactic]
        for key in group:
            if key.__contains__("."):
                parentKey = key.split(".")[0]
                parentValue = group[parentKey]
                oldValue = group[key]
                newValue = f"{parentValue}::{oldValue}"
                allResults[tactic][key] = newValue
    return allResults


rawData = requests.get("https://raw.githubusercontent.com/MBCProject/mbc-markdown/master/yfaq/mbc_matrix_with_ids.svg").content.decode("utf8")
rawData = rawData.replace("><", ">\n<")
cutData = cutByPhrases(rawData, "<g", "</g>")

allResults = {}
tacticGroup = {}
for chunk in cutData:
    if chunk.__contains__("technique"):
        idTag, name = extractTechnique(chunk)
        tacticGroup[idTag] = name

    elif chunk.__contains__("tactic-label"):
        tacticName = extractTactic(chunk)
        tacticName = tacticName.replace(" Micro-objective", "") # quick fix to remove "Micro-objective" bug in github svg file
        allResults[tacticName] = tacticGroup
        tacticGroup = {}
    else:
        pass

with open("MBCTagsList.tags", "w") as outFile:
    allResults = postProcessing(allResults)
    for tactic in allResults:
        group = allResults[tactic]
        for key in group:
            outFile.write(f"{tactic}::{group[key]} {key}\n")
