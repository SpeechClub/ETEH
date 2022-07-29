#!/bin/bash

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)

. ./path.sh

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1

stage=1
stop_stage=1

# The name of the total job
train_job_name=HKUST

# The training related config
task_file=eteh.lib.std_lm_task
task_name=LMTask
train_config=conf/lm/lm_baseline.yaml
data_config=conf/data/lm_data.yaml
exp_dir=exp/lm_baseline
train_epochs=30
random_seed=100

# The distributed training related config
init_method='tcp://127.0.0.1:8888'
total_gpu=1
total_node=1
node_idx=0
node_gpu=1


# The decoding related config
decode_conf=conf/decode/decode_ctc_att.yaml
char_list=../asr/data/$train_job_name/final_resource/dict.txt
decode_model=checkpoint.15
decode_data=data/HKUST/dev/feats.scp
decode_device=0
output_file=hkust-ctc-att.dev.txt

# You can train the LM in the ../lm
lm_model=None
lm_config=None

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	# This stage can generate all training data and config by Kaldi automaticly;
	# You can skip this stage and prepare these resource by yourself.
	# The generated config is decided by the data and the ${train_config}
	echo "Prepare the Training Data and Config"

	 ./local/prepare_hkust.sh data/$train_job_name data/$train_job_name/final_resource || exit 1

	# ./local/prepare_aishell.sh data/$train_job_name data/$train_job_name/final_resource
	# ./local/prepare_librispeech.sh data/$train_job_name data/$train_job_name/final_resource
	# ./local/prepare_swbd.sh data/$train_job_name data/$train_job_name/final_resource
	
	train_config=data/$train_job_name/final_resource/train_config.yaml
	data_config=data/$train_job_name/final_resource/data_config.yaml
	char_list=data/$train_job_name/final_resource/dict.txt
	
	echo "train_config is write in ${train_config}"
	echo "data_config is write in ${data_config}"
	echo "char_list is write in ${char_list}"
	# You can futher modify these file by yourself

fi	


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
		
	echo "Train the LM Model"	
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
		

