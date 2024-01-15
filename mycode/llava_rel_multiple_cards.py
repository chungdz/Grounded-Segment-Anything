import argparse
import os
import copy
import json
import sys
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import numpy as np

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

all_obj = set("person,broom,picture,closet,cabinet,blanket,window,table,paper,notebook,refrigerator,pillow,cup,glass,bottle,shelf,shoe,medicine,phone,camera,box,sandwich,book,bed,clothes,mirror,sofa,couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television".split(','))

# os.environ['TRANSFORMERS_CACHE'] = '/nobackup/users/bowu/model/transformers_cache'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_path', type=str, default='/home/azon/data/video/', help='path all images')
    parser.add_argument('--image_info_path', type=str, default='./objects.json', help='path all images')
    parser.add_argument('--output_path', type=str, default='llava.json', help='path to save annotations')
    args = parser.parse_args()

    print('load model')
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    print('load data')
    objects = json.load(open(args.image_info_path, 'r'))

    dlength = len(os.listdir(args.image_root_path))
    pbar = tqdm(os.walk(args.image_root_path), total=dlength, leave=True)
    for root, dirs, files in pbar:
        for name in tqdm(files, leave=False):
            image_path = os.path.join(root, name)
            pbar.set_description("Processing %s" % image_path)
            video_id = image_path.split('/')[-2].split('.')[-2]
            frame_id = image_path.split('/')[-1].split('.')[-2]
            labels = objects[video_id][frame_id]['labels']

            filtered = set()
            for item in labels:
                arr = item.split(' ')
                if arr[0] in all_obj:
                    filtered.add(arr[0])
            filtered = list(filtered)

            llava_prompt = "<image>\nFind important positional relations among only these objects: " + ','.join(filtered) + "\nASSISTANT:"
            image = Image.open(image_path)
            inputs = processor(text=llava_prompt, images=image, return_tensors="pt").to(0)
            generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
            return_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)[0]

            objects[video_id][frame_id]['llava'] = return_text

    
    json.dump(objects, open(args.output_path, 'w'))

# export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/
# python llava_rel.py --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava.json