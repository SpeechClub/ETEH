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
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symobles, e.g., <NOISE> etc.')
    parser.add_argument('token', type=str, help='filter list')
    parser.add_argument('trn', type=str, help='input file')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    text_file = args.token
    trn_file = args.trn
    space=args.space    
    if args.non_lang_syms is not None:
        with codecs.open(args.non_lang_syms, 'r', encoding="utf-8") as f:
            nls = [x.rstrip() for x in f.readlines()]
    else:
        nls=[]
    with open(text_file,'r',encoding='utf-8') as f:
        text = f.readlines()
    out_list = []
    for line in text:
        line = line.replace('\n','')
        name = line.split(' ')[0]
        txt = line.split(' ')[1:] 
        for i in range(len(txt)):
            if txt[i] in nls:
                txt[i] = '###'
                if i < len(txt)-1 and txt[i+1] == space:
                    txt[i+1] = '###' 
        while '###' in txt:
            txt.remove('###')
        new_line = ' '.join(txt) +" (%s)\n"%name
        out_list.append(new_line)
    with open(trn_file,'w',encoding='utf-8') as f:
        f.writelines(out_list)
if __name__ == "__main__":
    main()