#!/usr/bin/python3
import sys

mapping_dict = dict()
mapping_dict['aa'] = 'aa'
mapping_dict['ae'] = 'ae'
mapping_dict['ah'] = 'ah'
mapping_dict['ao'] = 'aa'
mapping_dict['aw'] = 'aw'
mapping_dict['ax'] = 'ah'
mapping_dict['ax-h'] = 'ah'
mapping_dict['axr'] = 'er'
mapping_dict['ay'] = 'ay'
mapping_dict['b'] = 'b'
mapping_dict['bcl'] = 'sil'
mapping_dict['ch'] = 'ch'
mapping_dict['d'] = 'd'
mapping_dict['dcl'] = 'sil'
mapping_dict['dh'] = 'dh'
mapping_dict['dx'] = 'dx'
mapping_dict['eh'] = 'eh'
mapping_dict['el'] = 'l'
mapping_dict['em'] = 'm'
mapping_dict['en'] = 'n'
mapping_dict['eng'] = 'ng'
mapping_dict['epi'] = 'sil'
mapping_dict['er'] = 'er'
mapping_dict['ey'] = 'ey'
mapping_dict['f'] = 'f'
mapping_dict['g'] = 'g'
mapping_dict['gcl'] = 'sil'
mapping_dict['h#'] = 'sil'
mapping_dict['hh'] = 'hh'
mapping_dict['hv'] = 'hh'
mapping_dict['ih'] = 'ih'
mapping_dict['ix'] = 'ih'
mapping_dict['iy'] = 'iy'
mapping_dict['jh'] = 'jh'
mapping_dict['k'] = 'k'
mapping_dict['kcl'] = 'sil'
mapping_dict['l'] = 'l'
mapping_dict['m'] = 'm'
mapping_dict['n'] = 'n'
mapping_dict['ng'] = 'ng'
mapping_dict['nx'] = 'n'
mapping_dict['ow'] = 'ow'
mapping_dict['oy'] = 'oy'
mapping_dict['p'] = 'p'
mapping_dict['pau'] = 'sil'
mapping_dict['pcl'] = 'sil'
mapping_dict['q'] = ''
mapping_dict['r'] = 'r'
mapping_dict['s'] = 's'
mapping_dict['sh'] = 'sh'
mapping_dict['t'] = 't'
mapping_dict['tcl'] = 'sil'
mapping_dict['th'] = 'th'
mapping_dict['uh'] = 'uh'
mapping_dict['uw'] = 'uw'
mapping_dict['ux'] = 'uw'
mapping_dict['v'] = 'v'
mapping_dict['w'] = 'w'
mapping_dict['y'] = 'y'
mapping_dict['z'] = 'z'
mapping_dict['zh'] = 'sh'



if len(sys.argv) != 2:
  print("!!Usage: python3 log2utt.py log voc")
  sys.exit(1)

voc=list()
with open("samples/data/timit_61.vocab") as f:
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
          a += mapping_dict[voc[int(char)]] + " "

        print(a.strip()+" ("+uttid+")")
        status = 0
