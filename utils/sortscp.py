#!/usr/bin/env python

# Apache 2.0
#!/usr/bin/env python

# Apache 2.0
import sys #传入3个参数，具体操作根据个人情况
import argparse
import codecs

def get_parser():
    parser = argparse.ArgumentParser(
        description='filter words in a text file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scp_in', type=str, help='input scp file')
    parser.add_argument('scp_out', type=str, help='input scp file')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    scp_in = args.scp_in
    scp_out = args.scp_out   
    with open(scp_in,'r',encoding='utf-8') as f:
        text = f.readlines()
    text.sort() 
    with open(scp_out,'w',encoding='utf-8') as f:
        f.writelines(text)
if __name__ == "__main__":
    main()