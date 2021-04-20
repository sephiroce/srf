import os
import sys
import numpy as np
feat_path = open(sys.argv[1])
status = 0
utt_id = ""
feats = ""

idx = 0
for line in feat_path:
  if status == 0 and "[" in line:
    idx += 1
    print(idx)
    utt_id = line.strip().split()[0]
    status = 1
  elif status == 1:
    feats += line.replace("]","").strip() + "\n"
    if "]" in line:
      with open(utt_id + ".npy.txt","w") as npy_file:
        npy_file.write(feats.strip())
      np.save(utt_id + ".npy", np.loadtxt(utt_id + ".npy.txt"))
      os.remove(utt_id + ".npy.txt")
      status = 0
      feats = ""
      utt_id = ""

feat_path.close()