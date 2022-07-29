#!/usr/bin/env python
import argparse
import os
import torch

def main():
    # last = sorted(args.snapshots, key=os.path.getmtime)
    last = args.snapshots
    if args.ids == 'last':
        last = last[-args.num:]
    else:
        ids = [int(i) for i in args.ids.split('_')]
        last = [last[i-1] for i in ids]
    print("average over", last)
    avg = None
    args.num=len(last)
    # sum
    for path in last:
        states = torch.load(path, map_location=torch.device("cpu"))
        if "model" in states:
            states = states["model"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k] ,args.num)
    torch.save(avg, args.out)

def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--ids", default='last', type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()

