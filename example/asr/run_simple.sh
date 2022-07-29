#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

stage=1
stop_stage=2

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
decode_model=checkpoint.99
decode_data=data/HKUST/dev/feats_cmvn.scp
decode_device=0 # -1 for using CPU
output_file=hyp.hkust_dev.txt

# You can train the LM in the ../lm
lm_model=../lm/exp/lm_baseline/checkpoint.24
lm_config=../lm/conf/lm/lm_baseline.yaml


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        
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
        

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "Decode with the ASR Model directly"
    mkdir -p ${exp_dir}/decode
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

