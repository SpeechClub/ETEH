#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

final_resource=$1
model_config_template=$2
feat_name=$3
use_ali=$4

train_set=`cat $final_resource/train_set`
valid_set=`cat $final_resource/valid_set`

token_dict=$final_resource/dict.txt

echo "Config Generation"
odim=$(cat $token_dict | wc -l) || exit 1
let odim=odim+2

bash local/conf_prep/generate_data_yaml.sh $final_resource/data_config.yaml \
        $train_set $valid_set $feat_name $odim $use_ali || exit 1

idim=$(feat-to-dim scp:${train_set}/$feat_name.scp -) || exit 1

sed $model_config_template -e "s/idim: ?/idim: ${idim}/" -e "s/size: ?/size: ${odim}/" \
        -e "s/odim: ?/odim: ${odim}/" -e "s/char_num: ?/char_num: ${odim}/" \
        -e "s#label_dict: ?#label_dict: '${token_dict}'#" \
        > $final_resource/train_config.yaml || exit 1
