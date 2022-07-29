#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

import editdistance
import sys
import re
res=re.compile("\[[^]]*\]")

ref=sys.argv[1]
hyp=sys.argv[2]

dref={}
for line in open(ref):
    line=line.strip().split(None,1)
    line[1]=res.sub("",line[1])
    dref['('+line[0]+')']=line[1]

totlen=0
toterr=0
for line in open(hyp):
    line=line.strip().rsplit(None,1)
    h,uid=("",line[0]) if len(line)==1 else line
    h=res.sub("",h).split()
    r=dref[uid].split()
    toterr+=editdistance.eval(r,h)
    totlen+=len(r)

print("error rate:",toterr/totlen)


