RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs_home'

### set gpus ###
# gpu_ids=0          # single-gpu
gpu_ids=0,1        # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    # NGPU=2
    # NGPU=3
    NGPU=2
    PORT=11769
    echo "HERE"
fi
################

### hyper params ###
lr=1e-4
min_lr=1e-6
update_learning_rate=0
warmup_epochs=40
epochs=4000
batch_size=64
ema_rate=0.999
ckpt_num=10
####################

### model stuff ###
note="release"
model='sdfusion_union_two_time_noise_octree'
df_cfg='configs/sdfusion_snet_union_2t.yaml'
# ckpt="logs_home/2024-02-01T17-12-18-sdfusion_hr_feature-snet-im_5-LR1e-4-release/ckpt/df_steps-latest.pth"

vq_model="GraphVAE"
vq_cfg='configs/shapenet_vae_lr.yaml'
# vq_ckpt="saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-9-00140.model.pth"
vq_ckpt="saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-8-00200.model.pth"

####################

### dataset stuff ###
dataset_mode='snet'
dataroot="data"
category='airplane'

#####################

### display & log stuff ###
display_freq=500
print_freq=25
save_steps_freq=3000
###########################


today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

name="${DATE_WITH_TIME}-${model}-${dataset_mode}-${category}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=3
    # batch_size=40
	max_dataset_size=120
    save_steps_freq=3
	display_freq=2
	print_freq=2
    name="DEBUG-${name}"
fi

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} \
            --lr ${lr} --epochs ${epochs}  --min_lr ${min_lr} --warmup_epochs ${warmup_epochs} --update_learning_rate ${update_learning_rate} --ema_rate ${ema_rate} \
            --model ${model} --df_cfg ${df_cfg} --ckpt_num ${ckpt_num} --category ${category} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} \
            --display_freq ${display_freq} --print_freq ${print_freq}
            --save_steps_freq ${save_steps_freq} \
            --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

echo "[*] Training with command: "

# if [ $multi_gpu = 1 ]; then
#     cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
# fi

if [ $multi_gpu = 1 ]; then

    cmd="--nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}

else

    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}

fi
