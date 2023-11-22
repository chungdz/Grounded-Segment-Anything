import argparse
import os
import copy
import json
import sys
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from collections import defaultdict

from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def get_annotations(image_path, 
                    text_prompt, 
                    groundingdino_model,
                    box_threshold=0.3, 
                    text_threshold=0.25):
    

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold
    )

    return {
        'image_path': image_path,
        'boxes': boxes.cpu().numpy().tolist(),
        'labels': phrases
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_path', type=str, default='/home/azon/data/video/', help='path all images')
    parser.add_argument('--output_path', type=str, default='./objects.json', help='path to save annotations')
    args = parser.parse_args()

    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    text_prompt = "person,broom,picture,closet/cabinet,blanket,window,table,paper/notebook,refrigerator,pillow,cup/glass/bottle,shelf,shoe,medicine,phone/camera,box,sandwich,book,bed,clothes,mirror,sofa/couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television"

    res = defaultdict(list)
    dlength = len(os.listdir(args.image_root_path))
    pbar = tqdm(os.walk(args.image_root_path), total=dlength, leave=True)
    for root, dirs, files in pbar:
        for name in tqdm(files, leave=False):
            image_path = os.path.join(root, name)
            pbar.set_description("Processing %s" % image_path)
            annotations = get_annotations(image_path, text_prompt, groundingdino_model)
            video_id = image_path.split('/')[-2].split('.')[-2]
            frame_id = image_path.split('/')[-1].split('.')[-2]
            res[video_id].append({
                'frame_id': frame_id,
                'boxes': annotations['boxes'],
                'labels': annotations['labels']
            })
    
    json.dump(res, open(args.output_path, 'w'))
