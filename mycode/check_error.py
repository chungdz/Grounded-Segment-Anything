import argparse
import os
import copy
import json
import sys
from tqdm import tqdm, trange

import numpy as np

import numpy as np
from collections import defaultdict

all_obj = set("person,broom,picture,closet,cabinet,blanket,window,table,paper,notebook,refrigerator,pillow,cup,glass,bottle,shelf,shoe,medicine,phone,camera,box,sandwich,book,bed,clothes,mirror,sofa,couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television".split(','))

parser = argparse.ArgumentParser()
parser.add_argument('--image_obj_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_', help='path all images')
parser.add_argument('--image_rel_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llama_result/llama_', help='path all images')
parser.add_argument('--image_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/error.json', help='path to save annotations')
parser.add_argument('--process_num', type=int, default=17)
args = parser.parse_args()

print('load data')
objects = json.load(open(args.image_info_path, 'r'))

print('gather llava')
llava_res = defaultdict(dict)
for index in trange(args.process_num):
    with open(args.image_obj_path + str(index) + '.txt', 'r') as f:
        obj_res = f.readlines()
        for line in obj_res:
            video_id, frame_id, return_text = json.loads(line)
            llava_res[video_id][frame_id] = return_text

print('gather llama')
llama_res = defaultdict(dict)
for index in trange(args.process_num):
    with open(args.image_rel_path + str(index) + '.txt', 'r') as f:
        rel_res = f.readlines()
        for line in rel_res:
            infodict = json.loads(line)
            video_id = infodict['video_id']
            frame_id = infodict['frame_id']
            relations = infodict['relations']
            llama_res[video_id][frame_id] = relations

# check if anything is missing
llava_missing_list = []
llama_missing_list = []
for video_id, video_info in objects.items():
    for frame_id, frame_info in video_info.items():
        if frame_id not in llava_res[video_id]:
            llava_missing_list.append((video_id, frame_id))
        if frame_id not in llama_res[video_id]:
            llama_missing_list.append((video_id, frame_id))

print('llava missing', llava_missing_list)
print('llama missing', llama_missing_list)

print('merge')
wrong_text_list = []
for video_id, video_info in objects.items():
    for frame_id, frame_info in video_info.items():
        if frame_id in llava_res[video_id] and frame_id in llama_res[video_id]:
            llava_text = llava_res[video_id][frame_id]
            llama_text = llama_res[video_id][frame_id]
            
            desc_index = llava_text.index(("ASSISTANT"))
            desc = llava_text[desc_index + 11:]
            try:
                struct_start = llama_text.index('{')
                struct_end = llama_text.index('}')
                rels = json.loads(llama_text[struct_start + 1: struct_end])
            except Exception as error:
                wrong_text_list.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "llava": llava_text,
                })

            frame_info['llava'] = desc
            frame_info['llama'] = rels

with open(args.output_path, 'w') as f:
    for item in wrong_text_list:
        f.write(json.dumps(item) + '\n')
    


