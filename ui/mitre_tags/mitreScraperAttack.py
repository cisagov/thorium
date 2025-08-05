#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests 
import pandas as pd
import json
import os
import mitreattack

os.system("python3 mitreattack-python/mitreattack/attackToExcel/attackToExcel.py")

df = pd.DataFrame(pd.read_excel("enterprise-attack/enterprise-attack.xlsx"))
ID = df["ID"]
name = df["name"]
tactic = df["tactics"]

# a brief description of what the tactic does, needs filtering
description = df["detection"]

tags = []
for i in range(len(ID)):
    tags.append(f"{tactic[i]}::{name[i]} {ID[i]}")

tags.sort()
outFile = open("attackTagsList.tags", "w")
for t in tags:
    outFile.write(f"{t}\n")
