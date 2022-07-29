
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import os
import sys

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description='split a json file for parallel processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str,default="../data.json",
                        help='json file')
    parser.add_argument('out', type=str,default="../data_out.json",
                        help='out json file')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # check directory
    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    # load json and split keys
    j = json.load(codecs.open(args.json, 'r', encoding="utf-8"))
    print(type(j['utts']))
    for key in j['utts']:
        print(key)
        print(j['utts'][key])
        data = j['utts'][key]
        data['input'][0].pop('name')
        data['output'][0].pop('name')
        data['output'][0].pop('text')
        data['output'][0].pop('token')
        data.pop("utt2spk")
    for key in j['utts']:
        print(j['utts'][key])

    json_str = json.dumps(j,indent=4,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(',', ': ')
                        )
    with open(args.out, 'w') as json_file:
        json_file.write(json_str)
                        
                        
