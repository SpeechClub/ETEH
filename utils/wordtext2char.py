#!/usr/bin/env python

# Apache 2.0
import sys #传入3个参数，具体操作根据个人情况
import argparse
import codecs

def get_parser():
    parser = argparse.ArgumentParser(
        description='filter words in a text file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('word', type=str, help='filter list')
    parser.add_argument('char', type=str, help='input file')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    text_file = args.word
    trn_file = args.char 
    with open(text_file,'r',encoding='utf-8') as f:
        text = f.readlines()
    out_list = []
    for line in text:
        line = line.replace('\n','')
        name = line.split(' ')[0]
        txt = line.split(' ')[1:] 
        txt = ''.join(txt)
        outs = name + " " + txt + "\n"
        out_list.append(outs)
    with open(trn_file,'w',encoding='utf-8') as f:
        f.writelines(out_list)
if __name__ == "__main__":
    main()