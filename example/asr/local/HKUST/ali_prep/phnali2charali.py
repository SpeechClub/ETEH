#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

import sys
import re
res=re.compile("[0-9]?_.*")
unk='SPN'
sil='SIL'

char_dict={}
for line in open(sys.argv[1]):
    line=line.split()
    #if line[0].isascii() or len(line[0])==1:
    #    if line[0] in char_dict:
    #        char_dict[line[0]].add(tuple(res.sub('',x) for x in line[2:]))
    #    else:
    #        char_dict[line[0]]={tuple(res.sub('',x) for x in line[2:])}
    #    continue
    #if all(x==line[0][0] for x in line[0][1:]):
    #    pron=[res.sub('',x) for x in line[2:]]
    #    pronlen=len(pron)//len(line[0])
    #    if all(pron[x*pronlen:(x+1)*pronlen] == pron[:pronlen] for x in range(1,len(line[0]))):
    #        if line[0][0] in char_dict:
    #            char_dict[line[0][0]].add(tuple(pron[:pronlen]))
    #        else:
    #            char_dict[line[0][0]]={tuple(pron[:pronlen])}
    #        continue
    if not line[0].isascii():
        consonant=True
        pron=[[]]
        for x in line[2:]:
            if not(len(x)>3 and x[-3].isdigit()):
                if consonant:
                    pron[-1].append(x)
                else:
                    consonant=True
                    pron.append([x])
            else:
                pron[-1].append(x)
                consonant=False
        if len(pron)==len(line[0]):
            for p,c in zip(pron,line[0]):
                if c in char_dict:
                    char_dict[c].add(tuple(res.sub('',x) for x in p))
                else:
                    char_dict[c]={tuple(res.sub('',x) for x in p)}
            continue

        if all(x==line[0][0] for x in line[0][1:]):
            pron=[res.sub('',x) for x in line[2:]]
            pronlen=len(pron)//len(line[0])
            if all(pron[x*pronlen:(x+1)*pronlen] == pron[:pronlen] for x in range(1,len(line[0]))):
                if line[0][0] in char_dict:
                    char_dict[line[0][0]].add(tuple(pron[:pronlen]))
                else:
                    char_dict[line[0][0]]={tuple(pron[:pronlen])}
                continue

    if line[0] in char_dict:
        char_dict[line[0]].add(tuple(res.sub('',x) for x in line[2:]))
    else:
        char_dict[line[0]]={tuple(res.sub('',x) for x in line[2:])}


def godeep(text,phones,already):
    if not text:
        if not phones:
            return already
        else:
            return False
    if not phones:
        return False
    if text[0] not in char_dict:
        if phones[0]==unk:
            return godeep(text[1:],phones[1:],already+[(unk,)])
        return False
    for pron in char_dict[text[0]]:
        if tuple(phones[:len(pron)])==pron:
            result=godeep(text[1:],phones[len(pron):],already+[pron])
            if result:
                return result
    return False

ali_dict={}
for line in open(sys.argv[2]):
    uttid,line=line.strip().split(None,1)
    line=[x.split() for x in line.split(';')]
    ali_dict[uttid]=list(zip(*line))

fst=open(sys.argv[4],'w')
fed=open(sys.argv[5],'w')
for line in open(sys.argv[3]):
    line=line.split()
    uttid,line=line[0],line[1:]
    if uttid not in ali_dict:
        continue
    #for x in line:
    #    if x.isascii():continue
    #    for c in x:
    #        if c not in char_dict:
    #            print(c,x)
    text=sum([[x] if x.isascii() or any(c not in char_dict for c in x) else list(x) for x in line],[])
#    for x in text:
#        if not x.isascii() and len(x)>1: print(x)
    #print(text)
    phones=[x for x in ali_dict[uttid][0] if x != sil]
    #print(phones)
    #print([char_dict[x] for x in text])
    result=godeep(text,phones,[])
    if result==False:
        print(uttid)
        print(text)
        print(phones)
        # print([char_dict[x] for x in text])
    else:
        fst.write(uttid)
        fed.write(uttid)
        i=0
        j=0
        st=0
        while i<len(ali_dict[uttid][0]):
            if ali_dict[uttid][0][i]==sil:
                st+=int(ali_dict[uttid][1][i])
                i+=1
            else:
                l=0
                for _ in range(len(result[j])):
                    l+=int(ali_dict[uttid][1][i])
                    i+=1
                if text[j][0]!='[' and text[j][0]!='<':
                    ts=[str(round(k*l/len(text[j])+st)) for k in range(len(text[j])+1)]
                else:
                    ts=[str(st),str(st+l)]
                fst.write(' '+' '.join(ts[:-1]))
                fed.write(' '+' '.join(ts[1:]))
                st+=l
                j+=1
        fst.write('\n')
        fed.write('\n')
fst.close()
fed.close()
