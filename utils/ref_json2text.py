#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert ASR json to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--remove-space", type=bool, default=False, help="remove space in text, true when use chinese")
    parser.add_argument("json", type=str, help="json files")
    parser.add_argument("dict", type=str, help="dict")
    parser.add_argument("ref", type=str, help="ref")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", args.json)
    with codecs.open(args.json, "r", encoding="utf-8") as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    with codecs.open(args.dict, "r", encoding="utf-8") as f:
        dictionary = f.readlines()
    char_list = [entry.split(" ")[0] for entry in dictionary]
    char_list.insert(0, "<blank>")
    char_list.append("<eos>")
    # print([x.encode('utf-8') for x in char_list])

    logging.info("writing ref text to %s", args.ref)
    r = codecs.open(args.ref, "w", encoding="utf-8")

    for x in j["utts"]:
        if "token" in j["utts"][x]["output"][0].keys():
            seq = [
                i for i in j["utts"][x]["output"][0]["token"].split()
            ]
        elif "tokenid" in j["utts"][x]["output"][0].keys():
            seq = [
                char_list[int(i)] for i in j["utts"][x]["output"][0]["tokenid"].split()
            ]
        if args.remove_space:
            r.write(x + " " + "".join(seq).replace("<eos>", "") + "\n")
        else:
            r.write(x + " " + " ".join(seq).replace("<eos>", "") + "\n")

