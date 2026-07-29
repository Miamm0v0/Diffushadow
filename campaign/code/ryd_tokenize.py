#!/usr/bin/env python3
"""R1: convert Rydberg occupation snapshots [delta, b1..bN] into the oseq
token layout [delta, P1,b1, ..., PN,bN] with every basis token fixed to Z
(P=4).  Occupation readout IS a Z-basis measurement, so this reuses the
entire Diffushadow training/generation stack with zero model changes.

Usage: python ryd_tokenize.py in1.json [in2.json ...]
Writes <name>_oseq.json next to each input.
"""
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    rows = json.load(open(path))
    out = []
    for row in rows:
        d, bits = row[0], row[1:]
        seq = [float(d)]
        for b in bits:
            seq += [4, int(b)]
        out.append(seq)
    dst = Path(path).with_name(Path(path).stem + "_oseq.json")
    json.dump(out, open(dst, "w"))
    print(f"{path} -> {dst}  ({len(out)} rows, N={(len(out[0]) - 1) // 2})")
