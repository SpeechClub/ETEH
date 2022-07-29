#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

import sys
import re
res=re.compile("[0-9]?_.*")
phone_dict={}
for line in open(sys.argv[1]):
    line=line.split()
    phone_dict[line[1]]=res.sub('',line[0])
fout=open(sys.argv[3],'w')
for line in open(sys.argv[2]):
    uttid,line=line.strip().split(None,1)
    line=map(lambda x:x.split(),line.split(';'))
    line=map(lambda x:phone_dict[x[0]]+' '+x[1],line)
    fout.write(uttid+' '+' ; '.join(line)+'\n')
fout.close()


