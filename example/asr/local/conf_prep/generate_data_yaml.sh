#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

data_conf=$1
train_dir=$2
valid_dir=$3
feat_name=$4
odim=$5
ali=$6

idim=$(feat-to-dim scp:${train_dir}/${feat_name}.scp -) || exit 1

if [ $ali == false ] ; then

cat <<EOF > $data_conf
clean_source:
  1:    
    text: 
        type: label_text
        path: ${train_dir}/token
        dim: $odim
    feats: 
        type: mat
        path: ${train_dir}/${feat_name}.scp
        length: ${train_dir}/utt2num_frames
        dim: $idim
    
valid_source:
  1:
    text: 
        type: label_text
        path: ${valid_dir}/token  
        dim: $odim
    feats: 
        type: mat
        path: ${valid_dir}/${feat_name}.scp
        length: ${valid_dir}/utt2num_frames
        dim: $idim
EOF

else

cat <<EOF > $data_conf
clean_source:
  1:
    text:
        type: label_text
        path: ${train_dir}/ali/token
        dim: $odim
    feats:
        type: mat
        path: ${train_dir}/ali/${feat_name}.scp
        length: ${train_dir}/ali/utt2num_frames
        dim: $idim
    y_beg:
        type: label_list
        path: ${train_dir}/ali/${ali}stt
    y_end:
        type: label_list
        path: ${train_dir}/ali/${ali}end

valid_source:
  1:
    text:
        type: label_text
        path: ${valid_dir}/ali/token
        dim: $odim
    feats:
        type: mat
        path: ${valid_dir}/ali/${feat_name}.scp
        length: ${valid_dir}/ali/utt2num_frames
        dim: $idim
    y_beg:
        type: label_list
        path: ${valid_dir}/ali/${ali}stt
    y_end:
        type: label_list
        path: ${valid_dir}/ali/${ali}end

EOF

fi
