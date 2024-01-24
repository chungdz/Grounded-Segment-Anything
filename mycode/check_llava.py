import os
from tqdm import tqdm, trange
import json
import numpy as np
import argparse
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

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
print(model.hf_device_map)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

print('load data')
objects = json.load(open(args.image_info_path, 'r'))
mapping = json.load(open('/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/mapping.json', 'r'))

# image = Image.open("/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/001YG.mp4/000089.png")
# prompt = "<image>\nFind important positional relations among only these objects: box,pillow,blanket,table,person,bed,bottle,clothes,phone,sofa,television,chair,laptop\nASSISTANT:"
# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
# image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)

video_id = '004QE'
frame_id = '000661'
print(mapping[video_id][frame_id])
labels = objects[video_id][frame_id]['labels']
image_path = os.path.join(args.image_root_path, video_id + '.mp4', frame_id + '.png')

filtered = set()
for item in labels:
    arr = item.split(' ')
    if arr[0] in all_obj:
        filtered.add(arr[0])
filtered = list(filtered)

p1 = '''<image>\nFind important relations among only these objects, do not use a new word to refer an object: '''
prompt = p1 + ','.join(filtered) + "\nASSISTANT:"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0)
generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)[0])
