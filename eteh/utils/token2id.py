import argparse
import codecs

def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--unk-id', '-u', default=1, type=int,
                        help='oov symbal will map to unk-id')
    parser.add_argument('token', type=str, nargs='?',
                        help='input token')

    parser.add_argument('token_dict', type=str, default=None, nargs='?',
                        help='input token_dict')

    return parser          

def main():
    parser = get_parser()    
    args = parser.parse_args()
    token_dict = {}
    with open(args.token_dict,encoding="utf-8") as f:
        dict_lines = f.readlines()
        for line in dict_lines:
            ll = line.split()
            token_dict[ll[0]] = ll[1]    

    if args.token:
        f = codecs.open(args.token, encoding="utf-8")
    else:
        exit()

    line = f.readline()
    while line:
        x = line.split()
        print(' '.join(x[:args.skip_ncols]), end=" ")
        a = x[args.skip_ncols:]
        a = [token_dict.setdefault(t,str(args.unk_id)) for t in a]
        print(' '.join(a))
        line = f.readline()
        
if __name__ == '__main__':
    main()