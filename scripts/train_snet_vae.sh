RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs'

### set gpus ###
# gpu_ids=0          # single-gpu
gpu_ids=4,5,6,7       # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    # NGPU=2
    # NGPU=3
    NGPU=4
    HOST_NODE_ADDR="localhost:27000"
    echo "HERE"
fi
################

### hyper params ###
lr=1e-3
min_lr=1e-6
update_learning_rate=0
warmup_epochs=40
epochs=300
batch_size=64
ema_rate=0.999
ckpt_num=5
seed=42
####################

### model stuff ###
model='vae'
df_yaml="octfusion_snet_uncond.yaml"
df_cfg="configs/${df_yaml}"
vq_model="GraphVAE"
vq_yaml="shapenet_vae_train.yaml"
vq_cfg="configs/${vq_yaml}"
vq_ckpt=""
mode="train_vae"

####################

### dataset stuff ###
note="vae"
dataset_mode='snet'
category='im_5'

#####################

### display & log stuff ###
display_freq=3000
print_freq=25
save_steps_freq=3000
save_latest_freq=500
###########################


today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

name="${category}_union/cascade_pretrain_${note}_lr${lr}"

debug=0
if [ "$mode" = "generate" ]; then
    df_cfg="${logs_dir}/${name}/${df_yaml}"
    vq_cfg="${logs_dir}/${name}/${vq_yaml}"
    ckpt="${logs_dir}/${name}/ckpt/df_steps-latest.pth"
fi

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --mode ${mode} \
            --lr ${lr} --epochs ${epochs}  --min_lr ${min_lr} --warmup_epochs ${warmup_epochs} --update_learning_rate ${update_learning_rate} --ema_rate ${ema_rate} --seed ${seed} \
            --model ${model} --df_cfg ${df_cfg} --ckpt_num ${ckpt_num} --category ${category} --split_dir ${split_dir} \
            --vq_model ${vq_model} --vq_cfg ${vq_cfg} \
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

    cmd="--nnodes=1 --nproc_per_node=${NGPU} --max-restarts=100 --rdzv-backend=c10d --rdzv-endpoint=${HOST_NODE_ADDR}  ${cmd}"
    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun ${cmd}

else

    echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} python3 ${cmd}

fi
