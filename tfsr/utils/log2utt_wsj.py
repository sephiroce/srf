#!/usr/bin/python3
import sys

if len(sys.argv) != 2:
  print("!!Usage: python3 log2utt.py log voc")
  sys.exit(1)

voc=list()
with open("samples/data/wsj.char.vocab") as f:
  for line in f:
    voc.append(line.strip())

status = 0
with open(sys.argv[1]) as f:
  for line in f:
    if status == 0:
      if "UTTID" in line:
        uttid = line.replace("UTTID: [\"","").replace("\"]","").strip()
        status = 1
    elif status == 1:
      if "values" in line:
        value = line.split("[")[2].split("]")[0]
        a = ""
        for char in value.split(" "):
          if voc[int(char)] == "<SPACE>":
            a += " "
          else:
            a += voc[int(char)]

        print(a.strip()+" ("+uttid+")")
        status = 0
