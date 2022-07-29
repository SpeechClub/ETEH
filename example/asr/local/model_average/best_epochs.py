#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

# Print the best "avg_num" epoch numbers with regard of "sort_key" using a training log file "log_input"

import sys

log_input=sys.argv[1]
sort_key=sys.argv[2] # e.g. "att_corr"
avg_num=int(sys.argv[3]) # e.g. 10
reverse=sys.argv[4] # e.g. True
reverse=True if reverse.lower()=="true" else False

dlist=[]
for line in open(log_input):
    if '{' in line:
        line=line.split(None,2)
        d=eval(line[2])
        d['epoch']=line[1][1:-1]
        dlist.append(d)
s=''
for j in sorted(dlist,key=lambda x:x[sort_key],reverse=reverse)[:avg_num]:
    s+=j['epoch']+','
print(s[:-1])

