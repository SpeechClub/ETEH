#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

ref_text=""
result_text=""
nlsyms=""
wer=false
utt2spk=""
bpe=""
bpemodel=""
remove_blank=true
filter=""
num_spkrs=1
help_message="Usage: $0 <data-dir>"

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1

if [ -n "${bpe}" ];then
  text2trn.py ${ref_text} ${dir}/ref.trn	
else
  if [ -n "${nlsyms}" ]; then
    text2token.py ${ref_text} --non-lang-syms ${nlsyms} --skip-ncols 1 > ${dir}/ref.token
    text2trn.py --non-lang-syms ${nlsyms} ${dir}/ref.token ${dir}/ref.trn
  else
    text2token.py ${ref_text}  --skip-ncols 1 > ${dir}/ref.token
    text2trn.py  ${dir}/ref.token ${dir}/ref.trn
  fi
fi


cp ${result_text} ${dir}/hyp.trn

if ${remove_blank}; then
  sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ -n "${nlsyms}" ]; then
  cp ${dir}/ref.trn ${dir}/ref.trn.org
  cp ${dir}/hyp.trn ${dir}/hyp.trn.org
  filt.py -v ${nlsyms} ${dir}/ref.trn.org > ${dir}/ref.trn
  filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi

if [ -n "${utt2spk}" ]; then
  echo "Add speaker"
  add_speaker.py ${dir}/ref.trn ${utt2spk}
  add_speaker.py ${dir}/hyp.trn ${utt2spk} 
fi

if [ -n "${filter}" ]; then
  sed -i.bak3 -f ${filter} ${dir}/hyp.trn
  sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi

sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i wsj -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
  if [ -n "$bpe" ]; then
	  spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
	  spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
  else	
	  sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
  fi
  sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt

  echo "write a WER result in ${dir}/result.wrd.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
