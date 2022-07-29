#!/usr/bin/env python

# Apache 2.0
import sys #传入3个参数，具体操作根据个人情况

def main(argv):
    res = argv[1]
    utt2spk = argv[2]
    with open(res,'r',encoding='utf-8') as fres:
        res_text = fres.readlines()
    #print(res_text)
    dict_temp = {}
    
    with open(utt2spk,'r',encoding='utf-8') as futt:
        for line in futt.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            dict_temp[k] = v
    #print(dict_temp)
    new_text = []
    for line in res_text:
        pos = line.find('(')
        name = line[pos:-1][1:-1]
        new_name = dict_temp[name]+'-'+name
        line = line[:pos] + '(' + new_name +')\n'
        new_text.append(line)
    with open(res,'w',encoding='utf-8') as fres:
        fres.writelines(new_text)
    
    
if __name__ == "__main__":
    main(sys.argv)
