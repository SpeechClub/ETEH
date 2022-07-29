#!/usr/bin/env python3
#coding:utf-8

import sys
for line in sys.stdin:
  blks = str.split(line)
  out_line = blks[0]
  for i in range(1, len(blks)):
    if blks[i] == "[VOCALIZED-NOISE]" or blks[i] == "[NOISE]" or blks[i] == "[LAUGHTER]":
      out_line += " " + blks[i]
      continue
    for j in blks[i]:
      out_line += " " + j
  print(out_line)
