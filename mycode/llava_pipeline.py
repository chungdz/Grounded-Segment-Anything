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
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline
from accelerate import PartialState
from torch.utils.data import Dataset, DataLoader

class VQADataset(Dataset):

    def __init__(self, image_info_path, image_root_path) -> None:
        super().__init__()
        self.objects = json.load(open(image_info_path, 'r'))
        self.root_path = image_root_path
        key_list = []
        for video_id, video_info in self.objects.items():
            for frame_id, frame_info in video_info.items():
                key_list.append((video_id, frame_id))
        self.key_list = key_list
    
    def __len__(self):
        return len(self.key_list)
    
    def __getitem__(self, index) -> dict:
        video_id, frame_id = self.key_list[index]
        image_path = os.path.join(self.root_path, video_id + '.mp4', frame_id + '.png')
        labels = self.objects[video_id][frame_id]['labels']

        filtered = set()
        for item in labels:
            arr = item.split(' ')
            if arr[0] in all_obj:
                filtered.add(arr[0])
        filtered = list(filtered)

        prompt = "<image>\nFind important positional relations among only these objects: " + ','.join(filtered) + "\nASSISTANT:"
        image = Image.open(image_path)

        return {'image': image, 'text': prompt, 'video_id': video_id, 'frame_id': frame_id}

    def collate_fn(self, batch):
        return {'text_list': [x['text'] for x in batch], 
                'image_list': [x['image'] for x in batch],
                'video_id': [x['video_id'] for x in batch], 
                'frame_id': [x['frame_id'] for x in batch]
                }

all_obj = set("person,broom,picture,closet,cabinet,blanket,window,table,paper,notebook,refrigerator,pillow,cup,glass,bottle,shelf,shoe,medicine,phone,camera,box,sandwich,book,bed,clothes,mirror,sofa,couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television".split(','))

parser = argparse.ArgumentParser()
parser.add_argument('--image_root_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/', help='path all images')
parser.add_argument('--image_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_', help='path to save annotations')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--process_num', type=int, default=16)
parser.add_argument('--rank', type=int, default=2)
parser.add_argument('--gpu_size', type=int, default=4)
args = parser.parse_args()

print('load data')
vqa = VQADataset(args.image_info_path, args.image_root_path)
dl = DataLoader(vqa, batch_size=args.batch_size * args.gpu_size, shuffle=False, num_workers=2, drop_last=False, collate_fn=vqa.collate_fn)

print('load pipeline')
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.tokenizer.padding_side = "left" 

distributed_state = PartialState()
model.to(distributed_state.device)

resf = open(args.output_path + '.txt', 'w')

for data in dl:
    
    with distributed_state.split_between_processes(data) as ddata:
        text_list = ddata['text_list']
        image_list = ddata['image_list']
        video_id_list = ddata['video_id']
        frame_id_list = ddata['frame_id']

        inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True).to(distributed_state.device)
        generate_ids = model.generate(**inputs, max_length=512, temperature=0.1, top_p=0.7, do_sample=True)
        returns = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_token0104zation_spaces=False)

        for video_id, frame_id, return_text in zip(video_id_list, frame_id_list, returns):
            resf.write(json.dumps([video_id, frame_id, return_text]) + '\n')

        resf.flush()
    
    break

resf.close()

# export HUGGINGFACE_HUB_CACHE=/nobackup/users/bowu/model/transformers_cache/
# CUDA_VISIBLE_DEVICES=0 python llava_rel.py --rank=0 --image_root_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/Charades_v1_480/ --image_info_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/objects.json --output_path=/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_