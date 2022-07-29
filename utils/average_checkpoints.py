#!/usr/bin/env python
import argparse
import os


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    if args.ids == 'last':
        last = last[-args.num:]
    elif args.ids == 'best':
        index = -1
        with open(args.log, "r") as fr:
            lines = fr.readlines()
        val_lines = []
        for line in lines:
            if "Valid_Att_Acc" in line:
                val_lines.append(line)
            else:
                continue
        val_scores = []
        for index, line in enumerate(val_lines, 0):
            val_scores.append([index, float(line.split()[2])])
        sorted_val_scores = sorted(val_scores, reverse=True, key=lambda x:x[1])
        last = [
            os.path.join(os.path.dirname(args.snapshots[0]), "checkpoint.%d" % (int(epoch[0])) )
            for epoch in sorted_val_scores[:args.num]
        ]
    else:
        ids = [int(i) for i in args.ids.split('_')]
        args.num = len(ids)
        last = [last[i-1] for i in ids]
    print("average over", last)
    avg = None

    if args.backend == 'pytorch':
        import torch
        # sum
        for path in last:
            states = torch.load(path, map_location=torch.device("cpu"))["model"]
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is None:
                continue
            if avg[k].dtype in (torch.int32, torch.int64, torch.uint8) :
                avg[k] //= args.num
            elif avg[k].dtype == torch.float32:
                avg[k] /= args.num
            else:
                print("Unknown datat type:", avg[k].dtype)
        avg = {'model':avg}
        torch.save(avg, args.out)
    elif args.backend == 'chainer':
        import numpy as np
        # sum
        for path in last:
            states = np.load(path)
            if avg is None:
                keys = [x.split('main/')[1] for x in states if 'model' in x]
                avg = dict()
                for k in keys:
                    avg[k] = states['updater/model:main/{}'.format(k)]
            else:
                for k in keys:
                    avg[k] += states['updater/model:main/{}'.format(k)]
        # average
        for k in keys:
            if avg[k] is not None:
                avg[k] /= args.num
        np.savez_compressed(args.out, **avg)
        os.rename('{}.npz'.format(args.out), args.out)  # numpy save with .npz extension
    else:
        raise ValueError('Incorrect type of backend')


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default='pytorch', type=str)
    parser.add_argument("--ids", default='last', type=str)
    parser.add_argument("--log", type=str, help="log file, required if 'ids' is 'best'")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
