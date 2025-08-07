import sys
import os


# line length greater than 5 is min requirement for containing the rule's name
# ie: "rule xyz : whatever"
def validRuleline(line):
    return len(line) > 5 and line[:5].__contains__("rule ")

def isPrivateRule(line):
    return len(line) > 8 and line[:8].__contains__("private")

# four different options here
# option 1 - rule xyz \n
# option 2 - rule xyz {\n
# option 3 - rule xyz : whatever \n
# option 4 - rule xyz : whatever {\n
def extractName(filename):
    rawData = open(filename, "r").read()
    split = rawData.split("\n")

    for s in split:
        if validRuleline(s):
            nameChunk = s.split()[1] #extracts rule name after splitting by space
            nameChunk = nameChunk.replace("{", "") #replace { with nothing in the case there is no second space
            return nameChunk
    return "ERROR"


def getMatchingFiles(name, nameMap):
    result = []
    for key in nameMap:
        value = nameMap[key]
        if value == name:
            result.append(key)
    return result

# returns the first insance of the name line
def getNameLine(rawData):
    split = rawData.split("\n")
    for s in split:
        if validRuleline(s):
            return s
    return "NO_NAME"

def alterName(f, name, counter):
    inFile = open(f, "r")
    rawData = inFile.read()
    inFile.close()

    nameLine = getNameLine(rawData)

    newNameLine = nameLine.replace(name, f"{name}_{counter}")
    newData = rawData.replace(nameLine, newNameLine)

    # do nothing if counter is 0
    if counter == 0:
        pass
    else:
        outFile = open(f, "w")
        outFile.write(newData)
        outFile.close()

# deletes all files with multiple yara rules in the same file
# multiple yara rules in the same file causes a bug with renaming
def deleteMultipleRules(filenames):
    for f in filenames:
        rawData = open(f, "r").read()
        split = rawData.split("\n")
        
        counter = 0
        for s in split:
            if validRuleline(s) or isPrivateRule(s):
                counter += 1

        if counter > 1:
            print(f"Removing Rule:{f}")
            os.system(f"rm {f}")

def pruneFilenames(filenames):
    prunedFilenames = []
    
    for f in filenames:
        if f.__contains__(".yar") and not f[0] == ".":
            prunedFilenames.append(f)
    return prunedFilenames

def getAllFiles(yaraDir):
    finalFilenames = []
    filenames = next(os.walk(yaraDir), (None, None, []))[2]
    for f in filenames:
        relPath = f"{yaraDir}/{f}"
        finalFilenames.append(relPath)
    return pruneFilenames(finalFilenames)

args = sys.argv
if not len(args) == 2:
    print("Use Case: python3 renameRules.py <yaraDirectoryToAlter>")
    exit(1)
else:
    print("Removing Bad Rules...")

yaraDir = args[1]
filenames = getAllFiles(yaraDir)
# delete files with multiple rules
deleteMultipleRules(filenames)
filenames = getAllFiles(yaraDir)


# create a mapping of all the names
nameMap = {}
for f in filenames:
    if f.__contains__(".yar"):
        nameMap[f] = extractName(f)

# get all unique rulenames
uniqueNames = list(dict.fromkeys(list(nameMap.values())))

finalRules = []
for name in uniqueNames:
    matchingFiles = getMatchingFiles(name, nameMap)
    counter = 0
    for f in matchingFiles:
        alterName(f, name, counter)
        counter += 1
    counter = 0
