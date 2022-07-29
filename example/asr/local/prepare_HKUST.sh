#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh
. ./cmd.sh

stage=0
stop_stage=3

hkust_audio_path= # /export/corpora/LDC/LDC2005S15/
hkust_text_path= # /export/corpora/LDC/LDC2005T32/

data=$1
final_resource=$2

load_kaldi_ali=$3

kaldi_lang_dir= # $KALDI_ROOT/egs/hkust/s5/data/lang
train_set_ali_dir= # $KALDI_ROOT/egs/hkust/s5/exp/tri5a_ali_train_nodup_sp
train_dev_ali_dir= # $KALDI_ROOT/egs/hkust/s5/exp/tri5a_ali_train_dev

log=log/HKUST

mkdir -p $data
mkdir -p $final_resource

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    bash local/HKUST/data_prep/hkust_data_prep.sh $hkust_audio_path $hkust_text_path $data
    bash local/HKUST/data_prep/hkust_format_data.sh $data
    for x in train dev; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" $data/${x}/wav.scp
    done
fi

train_set=train_nodup_sp
train_dev=train_dev
recog_set="dev train_dev"

echo $data/$train_set > $final_resource/train_set
echo $data/$train_dev > $final_resource/valid_set

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=$data/raw_fbank
    
    # make a dev set:
    # 4001 utts will be reduced to 4000 after feature extraction
    bash utils/subset_data_dir.sh --first $data/train 4001 $data/${train_dev} || exit 1
    bash utils/fix_data_dir.sh $data/${train_dev}
    n=$(($(wc -l < $data/train/segments) - 4001))
    bash utils/subset_data_dir.sh --last $data/train ${n} $data/train_nodev || exit 1

    # make a training set
    bash utils/data/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup || exit 1

    # speed-perturbed
    bash utils/data/perturb_data_dir_speed_3way.sh $data/train_nodup $data/train_nodup_sp || exit 1
    
    bash steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        --fbank-config conf/feature/fbank.conf --pitch-config conf/feature/pitch.conf \
	$data/${train_set} $log/make_fbank/${train_set} ${fbankdir} || exit 1
    bash utils/fix_data_dir.sh $data/${train_set}

    compute-cmvn-stats scp:$data/${train_set}/feats.scp $data/${train_set}/cmvn.ark
    
    perl utils/run.pl JOB=1:32 $log/apply_cmvn_${train_set}.JOB.txt apply-cmvn --norm-vars=true $data/${train_set}/cmvn.ark \
	    scp:${fbankdir}/raw_fbank_pitch_${train_set}.JOB.scp ark:- \| copy-feats --compress=true --compression-method=2  ark:- \
	    ark,scp:${fbankdir}/raw_fbank_pitch_cmvn_${train_set}.JOB.ark,${fbankdir}/raw_fbank_pitch_cmvn_${train_set}.JOB.scp || exit 1
    cat ${fbankdir}/raw_fbank_pitch_cmvn_${train_set}.{1..32}.scp > $data/${train_set}/feats_cmvn.scp
   
    for rtask in ${recog_set}; do
        bash steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            --fbank-config conf/feature/fbank.conf --pitch-config conf/feature/pitch.conf \
	    $data/${rtask} $log/make_fbank/${rtask} ${fbankdir} || exit 1
        bash utils/fix_data_dir.sh $data/${rtask}

	perl utils/run.pl JOB=1:10 $log/apply_cmvn_${rtask}.JOB.txt apply-cmvn --norm-vars=true $data/${train_set}/cmvn.ark \
            scp:${fbankdir}/raw_fbank_pitch_${rtask}.JOB.scp ark:- \| copy-feats --compress=true --compression-method=2  ark:- \
            ark,scp:${fbankdir}/raw_fbank_pitch_cmvn_${rtask}.JOB.ark,${fbankdir}/raw_fbank_pitch_cmvn_${rtask}.JOB.scp || exit 1
        cat ${fbankdir}/raw_fbank_pitch_cmvn_${rtask}.{1..10}.scp > $data/${rtask}/feats_cmvn.scp
    done
fi

token_dict=$final_resource/dict.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Token & Dictionary"
    for s in ${train_set} ${recog_set};do
        python3 local/HKUST/data_prep/hkust_segment_char.py < $data/$s/text > $data/$s/token || exit 1;
    done
    mkdir -p $final_resource
    cut -f 2- -d ' '  $data/$train_set/token > `dirname $token_dict`/tmp.txt
    for s in ${recog_set};do
        cut -f 2- -d ' '  $data/$s/token >> `dirname $token_dict`/tmp.txt
    done
    cat `dirname $token_dict`/tmp.txt | tr " " "\n" | sort | uniq | awk 'NF>0' | \
	    awk '{print $0 " " NR}' > $token_dict
    rm `dirname $token_dict`/tmp.txt
fi

if $load_kaldi_ali && [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Load Kaldi alignments"
    bash local/HKUST/ali_prep/aligz2charali.sh $kaldi_lang_dir $train_set_ali_dir $data/$train_set $data/${train_set}/ali
    bash local/HKUST/ali_prep/aligz2charali.sh $kaldi_lang_dir $train_dev_ali_dir $data/$train_dev $data/${train_dev}/ali
fi
