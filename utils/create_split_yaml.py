#!/usr/bin/env python

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(
        description='create yaml file containing splitted json files for parallel processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json', type=str,help='json file')
    parser.add_argument('--parts', '-p', type=int,help='Number of subparts to be prepared', default=0)
    parser.add_argument('--type', type=str,help='data type (no use)', default="TEST")
    parser.add_argument('--yaml', type=str,help='output yaml file')
    return parser

def create_yaml(json_file, data_parts, data_type, yaml_file):
    yaml_dict = {"clean_source": {}}
    json_file_head = json_file.split("data.json")[0]
    for index in range(data_parts):
        index = index + 1
        index_dict = {"type": data_type, "name": index, "path": json_file_head + "/split" + str(data_parts) + "utt/" + "data." + str(index) + ".json"}
        yaml_dict["clean_source"][index] =  index_dict
    with open(yaml_file, "w", encoding='utf-8') as fw:
        yaml.dump(yaml_dict, fw)


if __name__ == "__main__":
    args = get_parser().parse_args()
    create_yaml(args.json, args.parts, args.type, args.yaml)
    print("Complete creation of yaml file: " + args.yaml)
