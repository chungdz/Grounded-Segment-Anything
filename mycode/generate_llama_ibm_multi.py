import argparse
import os
import copy
import json
import sys
from tqdm import tqdm, trange
import re
from pprint import pprint

from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)


# from huggingface_hub import login
# login("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")
rel_list = "[on,behind,in_front_of,on_the_side_of,above,beneath,drinking_from,have_it_on_the_back,wearing,holding,lying_on,covered_by,carrying,eating,leaning_on,sitting_on,twisting,writing_on,standing_on,touching,wiping,at,under,near]"

parser = argparse.ArgumentParser()
# parser.add_argument('--frame_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_', help='path all images')
# parser.add_argument('--frame_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_filtered.txt', help='path all images')
# parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llama_result/llama_all.txt', help='path to save annotations')
parser.add_argument('--frame_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/filtered/0.txt', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llama_result/llama_0.txt', help='path to save annotations')
args = parser.parse_args()

load_dotenv()
client = Client(credentials=Credentials.from_env())

print('load data')
with open(args.frame_info_path, 'r') as f:
    all_frames = f.readlines()

print('load previous results')
respath = args.output_path
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

flen = len(all_frames)
prompt_list = []
res_dict_list = []
for findex in trange(sindex, flen):
    
    line = all_frames[findex]
    line_dict = json.loads(line)
    video_id = line_dict['video_id']
    frame_id = line_dict['frame_id']   
    desc = line_dict['desc']
    objstr = line_dict['objstr']

    prompt = '''USER:
    Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].

    Requirements:
    1. Only detect relations in target_relation list. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning. target_relation list: {}.
    2. Only detect subjects and objects in the given subject_object list. If the subject or object is not in the array, then discard the tuple.
    3. The output should contain at most ten tuples ranked by their importance. Person is more important.
    4. Do not infer or assume relations. Only depend on sentences themself.
    5. The Answer should be in valid Json format. All three words in tuples should be double quoted. Use brackets correctly. Use four stars at the beginning and end of the answer to indicate the start and end of the answer.

    For example:
    subject_object list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
    Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 
    Thinking guideline 1: All should contain three elements. Relations should separate from subjects and objects.
    Thinking guideline 2: Here is target_relation list: {}. It needs to be double checked to make sure all relations in the mid of the tuples should be exact the same as ones in the target_relation list.
    Thinking guideline 3: Here is subject_object list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]. Subjects and object at edge of the tuple should be exact the same as ones in the subject_object list.
    Thinking guideline 4: all relations should make sense, and person is more important.
    Thinking guideline 5: find the at most ten important tuples based on above thinkings and generate the final answer. The answer should in valid Json format. All three words in tuples should be double quoted. Use brackets correctly. Use four stars at the beginning and end of the answer to indicate the start and end of the answer.
    Answer:
    ****[["person", "sitting_on", "bed"],
    ["person", "near", "pillow"],
    ["person", "holding", "bag"],
    ["person", "wearing", "shoe"],
    ["cup", "on", "table"],
    ["bottle", "near", "cup"],
    ["book", "on", "table"],
    ["screen", "on", "table"],
    ["chair", "in_front_of", "table"]]****
    END

    Now give answer for the following subject_object list and input text. Based on the above example and requirements, generate the same thinkings guidelines in the example. And then generate the final answer in Json format with four stars at the start and end. Use brackets correctly. Do not over thinking too much.

    subject_object list: [{}]
    Input text:{}
    ASSISTENT:'''.format(rel_list, rel_list, objstr, desc)
    
    resdict = {
        'video_id': video_id,
        'frame_id': frame_id,
    }
    
    prompt_list.append(prompt)
    res_dict_list.append(resdict)

failed_encode = 0
plen = len(prompt_list)
batch_size = 400
for pindex in trange(0, 2000, batch_size):

    prompt_list_batch = prompt_list[pindex: pindex + batch_size]
    res_dict_list_batch = res_dict_list[pindex: pindex + batch_size]

    for idx, response in tqdm(enumerate(
            client.text.generation.create(
                model_id="meta-llama/llama-2-70b-chat",
                inputs=prompt_list_batch,
                parameters=TextGenerationParameters(
                    max_new_tokens=2048,
                    min_new_tokens=20,
                    temperature=0.1,
                    top_p=0.7,
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                    ),
                ),
            )
        ), total=len(prompt_list_batch), leave=False):

        result = response.results[0]
        res_dict = res_dict_list_batch[idx]
        res_dict['input'] = result.input_text
        res_dict['res'] = result.generated_text

        em = None
        rels = []
        try:
            answer = result.generated_text
            startp = [m.start() for m in re.finditer("\*\[", answer)][-1]
            endp = [m.start() for m in re.finditer("]\*", answer)][-1]
            strlist = answer[startp + 1: endp + 1]
            rels = json.loads(strlist)
        except Exception as e:
            em = e

        resdict['rels'] = rels
        resf.write(json.dumps(resdict) + '\n')
        resf.flush()

        if len(rels) == 0:
            print(prompt_list_batch[idx])
            print(result.generated_text)
            print(em)
            failed_encode += 1  
            print('rels', rels)
    
    assert(idx == batch_size - 1)

print('failed_encode', failed_encode)
resf.close()
