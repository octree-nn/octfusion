RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### set gpus ###
gpu_ids=0,1,2,3       # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    NGPU=4
    HOST_NODE_ADDR="localhost:25000"
    echo "HERE"
fi
################

### model stuff ###
model='union_2t'
mode="$1"
stage_flag="$2"
dataset_mode='snet'
note="test"
category="$3"

df_yaml="octfusion_${dataset_mode}_uncond.yaml"
df_cfg="configs/${df_yaml}"
vq_model="GraphVAE"
vq_yaml="vae_${dataset_mode}_eval.yaml"
vq_cfg="configs/${vq_yaml}"
vq_ckpt="saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth"

### hyper params ###
lr=2e-4
min_lr=1e-6
update_learning_rate=0
warmup_epochs=40
ema_rate=0.999
ckpt_num=3
seed=42

if [ $stage_flag = "lr" ]; then
    epochs=3000
    batch_size=16
else
    epochs=500
    batch_size=2
fi

if [ $mode = "train" ]; then
    pretrain_ckpt="saved_ckpt/diffusion-ckpt/${category}/df_steps-split.pth"
else
    ckpt="saved_ckpt/diffusion-ckpt/${category}/df_steps-union.pth"
fi

####################

#####################

### display & log stuff ###
display_freq=1000
print_freq=25
save_steps_freq=3000
save_latest_freq=500
###########################


today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

name="${category}_union/${model}_${note}_lr${lr}"

debug=0

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --mode ${mode} \
    --lr ${lr} --epochs ${epochs}  --min_lr ${min_lr} --warmup_epochs ${warmup_epochs} --update_learning_rate ${update_learning_rate} --ema_rate ${ema_rate} --seed ${seed} \
    --model ${model} --stage_flag ${stage_flag} --df_cfg ${df_cfg} --ckpt_num ${ckpt_num} --category ${category} \
    --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --save_steps_freq ${save_steps_freq} --save_latest_freq ${save_latest_freq} \
    --debug ${debug}"

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi
if [ ! -z "$pretrain_ckpt" ]; then
    cmd="${cmd} --pretrain_ckpt ${pretrain_ckpt}"
    echo "second stage training with pretrain_ckpt=${pretrain_ckpt}"
fi
if [ ! -z "$split_dir" ]; then
    cmd="${cmd} --split_dir ${split_dir}"
    echo "generate with split_dir=${split_dir}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

echo "[*] Training with command: "

if [ $multi_gpu = 1 ]; then

    cmd="--nnodes=1 --nproc_per_node=${NGPU} --rdzv-backend=c10d --rdzv-endpoint=${HOST_NODE_ADDR}  ${cmd}"
    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}

else

    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}

fi
