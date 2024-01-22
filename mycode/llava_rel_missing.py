import argparse
import os
import copy
import json
import sys
from tqdm import tqdm, trange

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import numpy as np

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

all_obj = set("person,broom,picture,closet,cabinet,blanket,window,table,paper,notebook,refrigerator,pillow,cup,glass,bottle,shelf,shoe,medicine,phone,camera,box,sandwich,book,bed,clothes,mirror,sofa,couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television".split(','))

parser = argparse.ArgumentParser()
parser.add_argument('--image_root_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/', help='path all images')
parser.add_argument('--image_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_', help='path to save annotations')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--process_num', type=int, default=16)
parser.add_argument('--rank', type=int, default=16)
parser.add_argument('--sindex', type=int, default=0)
args = parser.parse_args()

print('load model')
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model.to(0)

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
print('load data')
objects = json.load(open(args.image_info_path, 'r'))
key_list = []
for video_id, video_info in objects.items():
    for frame_id, frame_info in video_info.items():
        key_list.append((video_id, frame_id))

key_list = [('ZZXQF', '000312'), ('ZZXQF', '000317'), ('ZZXQF', '000349'), ('ZZXQF', '000364'), ('ZZXQF', '000367'), ('ZZXQF', '000387'), ('ZZXQF', '000394'), ('ZZXQF', '000416'), ('ZZXQF', '000438'), ('ZZXQF', '000468'), ('ZZXQF', '000482'), ('ZZXQF', '000507'), ('ZZXQF', '000520'), ('ZZXQF', '000628')]

resf = open(args.output_path + str(args.rank) + '.txt', 'a')
pbar = trange(len(key_list) // args.batch_size + 1)

for _ in pbar:
    
    image_list = []
    prompt_list = []
    batch_list = []
    for i in range(args.batch_size):
        if len(key_list) == 0:
            break
        video_id, frame_id = key_list.pop()
        image_path = os.path.join(args.image_root_path, video_id + '.mp4', frame_id + '.png')
        pbar.set_description("Processing %s" % image_path)
    
        labels = objects[video_id][frame_id]['labels']

        filtered = set()
        for item in labels:
            arr = item.split(' ')
            if arr[0] in all_obj:
                filtered.add(arr[0])
        filtered = list(filtered)

        prompt_list.append("<image>\nFind important positional relations among only these objects: " + ','.join(filtered) + "\nASSISTANT:")
        image_list.append(Image.open(image_path))
        batch_list.append((video_id, frame_id))
    
        inputs = processor(text=prompt_list, images=image_list, return_tensors="pt", padding=True).to(0)
        generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
        returns = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)

    for (video_id, frame_id), return_text in zip(batch_list, returns):
        resf.write(json.dumps([video_id, frame_id, return_text]) + '\n')

    resf.flush()

resf.close()

# export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/
# CUDA_VISIBLE_DEVICES=0 python llava_rel.py --rank=0 --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_