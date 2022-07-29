#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import re
import sys

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rate', '-u', default=0.5, type=float,
                        help='max unk rate')
    parser.add_argument('--unk', default='<unk>', type=str,
                        help='unk symbol')
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    parser.add_argument('dict', type=str, default=False, nargs='?',
                        help='input dictionary')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout if is_python2 else sys.stdout.buffer)
    token_dict = dict_reader(args.dict)
    line = f.readline()
    out_list=[]
    clear_list=[]
    while line:
        x = line.split()
        unk_count=0
        for i in range(len(x)):
            if x[i] in token_dict:
                continue
            else:
               x[i] = args.unk
               unk_count = unk_count + 1
        new_line = ' '.join(x) + '\n'            
        if unk_count / len(x) > args.rate:
            clear_list.append(line)
        else:
            out_list.append(new_line)
        line = f.readline()
    with open('clear_'+args.text,'w',encoding='utf-8') as fres:
        fres.writelines(out_list)
    with open('disgard_'+args.text,'w',encoding='utf-8') as fres:
        fres.writelines(clear_list)
        
def dict_reader(path,sc=' ',append=True,eos='<eos>'):
    world_dict = {}
    last = 0
    with open(path,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()
    for line in lines:
        key, value = line.split(sc)[0], int(line.split(sc)[1])
        world_dict[key] = value
        last = value + 1
    if append:
        world_dict[eos] = last
    return world_dict


if __name__ == '__main__':
    main()
