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
parser.add_argument('--rank', type=int, default=2)
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

klen = len(key_list)
rank_num = (klen // args.process_num) + 1
rank_start = rank_num * args.rank
rank_end = rank_num * (args.rank + 1)
print('all length', klen, 'rank_start', rank_start, 'rank_end', rank_end)
key_list = key_list[rank_start + args.sindex: rank_end]

print('load previous results')
respath = args.output_path + str(args.rank) + '.txt'
if os.path.exists(respath):
    with open(respath, 'r') as f:
        res = f.readlines()
    resf = open(respath, 'w')
    print('previous results length', len(res), 'drop last line and start from there')
    # avoid broken line at the end
    for line in res[:-1]:
        resf.write(line)
    sindex = len(res) - 1
else:
    print('previous results not exist, start from 0')
    sindex = 0
    resf = open(respath, 'w')

for video_id, frame_id in tqdm(key_list):
    
    image_path = os.path.join(args.image_root_path, video_id + '.mp4', frame_id + '.png')    
    labels = objects[video_id][frame_id]['labels']

    filtered = set()
    for item in labels:
        arr = item.split(' ')
        if arr[0] in all_obj:
            filtered.add(arr[0])
    filtered = list(filtered)

    obstr = ','.join(filtered)

    prompt = '''<image>\n Describe all important relations with natural language only using the objects in the list: [{}]. 
    ASSISTANT:'''.format(obstr)
    
    image = Image.open(image_path)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0)
    generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
    desc_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)[0]

    desc_index = desc_text.index(("ASSISTANT"))
    desc = desc_text[desc_index + 11:]

    resdict = {
        'video_id': video_id,
        'frame_id': frame_id,
        'desc': desc,
        'objstr': obstr
    }

    resf.write(json.dumps(resdict) + '\n')

    resf.flush()

resf.close()

# export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/
# CUDA_VISIBLE_DEVICES=0 python llava_rel.py --rank=0 --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_