#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

stage=0
stop_stage=4

# The name of the total job
train_job_name=HKUST

# The training related config
task_file=eteh.lib.std_asr_task
task_name=CtcAttTask_Kaldi
train_config_template=conf/e2e/eteh_baseline.yaml
train_config=data/$train_job_name/final_resource/train_config.yaml
data_config=data/$train_job_name/final_resource/data_config.yaml
exp_dir=exp/$train_job_name/$(basename ${train_config_template%.*})
train_epochs=100
random_seed=100

# The distributed training related config
init_method='tcp://127.0.0.1:8888'
total_gpu=4
total_node=1
node_idx=0
node_gpu=4

# The decoding related config
decode_conf=conf/decode/decode_ctc_att.yaml
char_list=data/$train_job_name/final_resource/dict.txt
decode_model=model.attaccbest10avg
decode_data=data/HKUST/dev/feats_cmvn.scp
decode_device=0 # -1 for using CPU
output_file=hyp.hkust_dev.txt
ref_file=data/HKUST/dev/text

# You can train the LM in the ../lm
lm_model=../lm/exp/lm_baseline/checkpoint.24
lm_config=../lm/conf/lm/lm_baseline.yaml

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # This stage can generate all training data by Kaldi automaticly;
    # You can skip this stage and prepare these resource by yourself.
    echo "Prepare Training Data"
    
    # Set prepare_ali true if you want to use Kaldi alignments for customized model training
    prepare_ali=false

    bash ./local/prepare_$train_job_name.sh data/$train_job_name data/$train_job_name/final_resource $prepare_ali || exit 1

    echo "char_list is written in ${char_list}"
    # You can futher modify these file by yourself

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # This stage can generate all training config automaticly;
    # You can skip this stage and prepare these resource by yourself.
    # The generated config is decided by the data and ${train_config_template}
    echo "Prepare Config"
    
    # Set use_ali "token" or "word" if you want to use token/word level alignments for customized model training
    use_ali=false

    bash ./local/prepare_config.sh data/$train_job_name/final_resource $train_config_template feats_cmvn $use_ali || exit 1

    echo "train_config is written in ${train_config}"
    echo "data_config is written in ${data_config}"
    # You can futher modify these file by yourself

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        
    echo "Train the ASR Model"    
    mkdir -p ${exp_dir}
    python ${PLAT_ROOT}/bin/train_dist.py -train_config ${train_config} \
        -data_config ${data_config} \
        -train_name ${train_job_name} \
        -task_file ${task_file} \
        -num_gpu ${total_gpu} \
        -task_name ${task_name} \
        -exp_dir ${exp_dir} \
        -dist_method nccl \
        -num_epochs ${train_epochs} \
        -init_method ${init_method} \
        -seed ${random_seed} \
        -nodes ${total_node} \
        -gpus ${node_gpu} \
        -node_id ${node_idx}

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo "Model average"
	decode_model=model.attaccbest10avg
    best10=`python3 local/model_average/best_epochs.py ${exp_dir}/rank0_train.log att_corr 10 True`
    eval python3 local/model_average/average_checkpoints.py \
                --snapshots ${exp_dir}/checkpoint.{${best10}} \
                --out ${exp_dir}/${decode_model} \
                --num 10 \
                --ids last

    echo "Decode with the ASR Model"

    mkdir -p ${exp_dir}/decode
        
    mp=true # if true: use multiprocess decoding (on CPU)
    if $mp ; then
        nj=32
        eval mkdir -p `dirname ${decode_data}`/split$nj/{1..${nj}}
        eval perl utils/split_scp.pl ${decode_data} `dirname ${decode_data}`/split$nj/{1..${nj}}/`basename ${decode_data}`
        for j in `seq 1 $nj`;do
            python3 ${PLAT_ROOT}/bin/decode.py \
                -model ${exp_dir}/${decode_model} \
                -model_config ${train_config} \
                -lm_model ${lm_model} \
                -lm_config ${lm_config} \
                -decode_config ${decode_conf} \
                -data_list `dirname ${decode_data}`/split$nj/$j/`basename ${decode_data}` \
                -output_file ${exp_dir}/decode/${output_file}.$j \
                -char_list ${char_list} \
                -gpu -1 &
        done
        wait
        eval cat ${exp_dir}/decode/${output_file}.{1..${nj}} > ${exp_dir}/decode/${output_file}
    else
        python3 ${PLAT_ROOT}/bin/decode.py \
            -model ${exp_dir}/${decode_model} \
            -model_config ${train_config} \
            -lm_model ${lm_model} \
            -lm_config ${lm_config} \
            -decode_config ${decode_conf} \
            -data_list ${decode_data} \
            -output_file ${exp_dir}/decode/${output_file} \
            -char_list ${char_list} \
            -gpu ${decode_device}
    fi
	
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # You can skip this stage and do scoring by yourself.
    echo "Score with Sclite"
    mkdir -p ${exp_dir}/decode/score
    paste -d ' ' <(cut -f 2- -d ' ' ${ref_file} | sed -r "s/\[[^]]*\]//g" | sed "s/ //g") <(awk '{print "("$1")"}' ${ref_file}) | iconv -f utf8 -t cp936 > ${exp_dir}/decode/score/ref.txt
    sed -r "s/\[[^]]*\]//g" ${exp_dir}/decode/${output_file} | sed "s/ //g" | sed "s/(/ (/"  | iconv -f utf8 -t cp936 > ${exp_dir}/decode/score/hyp.txt
    local/scoring/sclite -r ${exp_dir}/decode/score/ref.txt -h ${exp_dir}/decode/score/hyp.txt -i wsj -o all -c NOASCII -e gb
    cat ${exp_dir}/decode/score/hyp.txt.sys
fi

