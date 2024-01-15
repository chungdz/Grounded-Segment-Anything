# Local device
python generate_bbox.py \
    --image_root_path=/mnt/d/data/Charades_v1_480 \
    --output_path=object_new.json

export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/

# 16 processes at Satori
python llava_rel.py \
        --rank=0 \
        --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ \
        --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json \
        --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_
