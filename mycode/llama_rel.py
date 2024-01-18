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
from transformers import LlamaTokenizer, LlamaForCausalLM

from huggingface_hub import login
login("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")

all_obj = set("person,broom,picture,closet,cabinet,blanket,window,table,paper,notebook,refrigerator,pillow,cup,glass,bottle,shelf,shoe,medicine,phone,camera,box,sandwich,book,bed,clothes,mirror,sofa,couch,floor,bag,dish,laptop,door,towel,food,chair,doorknob,doorway,groceries,hands,light,vacuum,television".split(','))
p1 = '''
Generate tuples [Entity, Relation, Entity] from quoted text which is describing the image. The tuples show most important relations between two entities in the sentences. 

Requirements:
1. The relation are limited in following array: [on, behind, in_front_of, on_the_side_of, above, beneath, drinking_from, have_it_on_the_back, wearing, holding, lying_on, covered_by, carrying, eating, leaning_on, sitting_on, twisting, writing_on, standing_on, touching, wiping, at, under, near] You should only use relations I give you. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning.
2. Do not infer or assume relations. Only depend on sentences themself.
3. Use a valid Json format to form the output. Start and end with curly brackets.
4. The relation should make sense.
5. The output should contain at most 15 tuples

For example:

Input test: "In the image, the woman sits on the bed near a pillow".
Output: {[["woman", "on", "bed"],
        ["woman", "near", "pillow"]]}

Now, generate tuples of relation for me with this input text:
'''


parser = argparse.ArgumentParser()
parser.add_argument('--frame_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llama_result/llama_', help='path to save annotations')
parser.add_argument('--rank', type=int, default=2)
parser.add_argument('--sindex', type=int, default=0)
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')

print('load data')
with open(args.frame_info_path + str(args.rank) + '.txt', 'r') as f:
    all_frames = f.readlines()
resf = open(args.output_path + str(args.rank) + '.txt', 'a')

flen = len(all_frames)
for findex in trange(args.sindex, flen):
    
    line = all_frames[findex]
    video_id, frame_id, desc_text = json.loads(line)
    desc_index = desc_text.index(("ASSISTANT"))
    desc = desc_text[desc_index + 11:]
    prompt = p1 + '"{}"\nASSISTANT:'.format(desc)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(0)
    generation_output = model.generate(input_ids=input_ids, max_length=2048, temperature=0.1, top_p=0.7, do_sample=True)
    res = tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer_index = res.index("ASSISTANT")
    answer = res[answer_index + 11:]
    
    resdict = {
        'video_id': video_id,
        'frame_id': frame_id,
        'desc_text': desc_text,
        'relations': answer,
    }
    resf.write(json.dumps(resdict) + '\n')

    resf.flush()

    if findex == 10:
        break

resf.close()