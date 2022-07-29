#!/bin/bash
#

if [ -f ./path.sh ]; then . ./path.sh; fi

data=$1
mkdir -p $data/train $data/dev

# Copy stuff into its final locations...

for f in wav.scp text segments reco2file_and_channel; do
  cp $data/local/train/$f $data/train/$f || exit 1;
done

awk '{print $1,$1}' $data/train/text > $data/train/utt2spk
awk '{print $1,$1}' $data/train/text > $data/train/spk2utt

for f in wav.scp text segments reco2file_and_channel; do
  cp $data/local/dev/$f $data/dev/$f || exit 1;
done

awk '{print $1,$1}' $data/dev/text > $data/dev/utt2spk
awk '{print $1,$1}' $data/dev/text > $data/dev/spk2utt

echo hkust_format_data succeeded.
