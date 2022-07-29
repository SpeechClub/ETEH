#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

kaldi_lang_dir=$1 # /data/yangrunyan/kaldi_alis/hkust/lang

ali_dir=$2 # /data/yangrunyan/kaldi_alis/hkust/tri5a_ali_train_dev
data_dir=$3 # =data/HKUST/train_dev
data_dir_ali=$4 # data/HKUST/train_dev_ali

nj=`ls $ali_dir/ali.*.gz | wc -l`
for j in `seq 1 $nj` ;do
    (ali-to-phones --write-lengths $ali_dir/final.mdl ark:"gunzip -c $ali_dir/ali.$j.gz |" ark,t:$ali_dir/ali.$j.int
    python3 local/HKUST/ali_prep/aliint2aliphn.py $kaldi_lang_dir/phones.txt $ali_dir/ali.$j.int $ali_dir/ali.$j.txt ) &
done
wait

eval cat $ali_dir/ali.{1..${nj}}.txt > $ali_dir/ali.txt

mkdir -p $data_dir_ali

python3 local/HKUST/ali_prep/phnali2charali.py $kaldi_lang_dir/phones/align_lexicon.txt $ali_dir/ali.txt $data_dir/text $data_dir_ali/tokenstt $data_dir_ali/tokenend

ln -s tokenstt $data_dir_ali/wordstt
ln -s tokenend $data_dir_ali/wordend

for f in feats.scp feats_cmvn.scp text token utt2num_frames;do
    perl utils/filter_scp.pl $data_dir_ali/tokenstt $data_dir/$f > $data_dir_ali/$f
done


