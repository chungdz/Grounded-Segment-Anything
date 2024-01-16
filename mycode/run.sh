# Local device
python generate_bbox.py \
    --image_root_path=/mnt/d/data/Charades_v1_480 \
    --output_path=object_new.json

export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/

# 16 processes at Satori
CUDA_VISIBLE_DEVICES=0 python llava_rel.py \
                        --rank=0 \
                        --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ \
                        --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json \
                        --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_

srun --gres=gpu:1 -n 32 --mem=100G  --time 24:00:00  --pty /bin/bash

conda activate llava
cd /nobackup/users/bowu/code/Reasoning_vlt/workspace
export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/

ssh -X node0005
CUDA_VISIBLE_DEVICES=0 python llava_rel.py --rank=0 --sindex=0 (llava-7)
CUDA_VISIBLE_DEVICES=1 python llava_rel.py --rank=1 --sindex=0  (llavasub5)
CUDA_VISIBLE_DEVICES=2 python llava_rel.py --rank=2 --sindex=0  (llavasub10)
CUDA_VISIBLE_DEVICES=3 python llava_rel.py --rank=3 --sindex=0  (llavasub30)

ssh -X node0007
CUDA_VISIBLE_DEVICES=0 python llava_rel.py --rank=4 --sindex=0 (llavasub20)
CUDA_VISIBLE_DEVICES=1 python llava_rel.py --rank=5 --sindex=0 (llavasub4-260)
CUDA_VISIBLE_DEVICES=2 python llava_rel.py --rank=6 --sindex=0 (llava6-270)
CUDA_VISIBLE_DEVICES=3 python llava_rel.py --rank=7 --sindex=0 (llavasub70)

