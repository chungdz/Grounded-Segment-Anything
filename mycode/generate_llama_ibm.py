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
parser.add_argument('--frame_info_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llava_result/llava_filtered.txt', help='path all images')
parser.add_argument('--output_path', type=str, default='/nobackup/users/bowu/data/STAR/Raw_Videos_Frames/llama_result/llama_all.txt', help='path to save annotations')
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
for findex in trange(sindex, 20):
    
    line = all_frames[findex]
    line_dict = json.loads(line)
    video_id = line_dict['video_id']
    frame_id = line_dict['frame_id']   
    desc = line_dict['desc']
    objstr = line_dict['objstr']

    prompt = '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:
    Task: detect relations between subjects and objects from input text and generate Json formatted tuple list [Subject, Relation, Object].
    Requirements:

    1. Only detect relations in target_relation list. If the relation is not in the array, then discard the tuple or try to find relation with similar meaning. target_relation list: {}.
    2. Only detect subjects and objects in the given subject_object list. If the subject or object is not in the array, then discard the tuple.
    3. The output should contain at most 8 tuples ranked by their importance. Person is more important.
    4. Do not infer or assume relations. Only depend on sentences themself.

    For example:

    subject_object list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]
    Input text: "In the image, the woman sits on the bed near by a pillow. The women's hands is holding the bag. The woman is wearing a blue shirt and white shoes. There is a cup on the table, and a bottle is placed nearby. A book and a monitor is also present on the table. A chair is in the room and in the front of the table". 

    Thinking 1: Guideline: All should contain three elements. Relations should separate from subjects and objects.
    Thinking 2: Guideline: Here is target_relation list: {}. It needs to be double checked to make sure all relations in the mid of the tuples should be exact the same as ones in the target_relation list.
    Thinking 3: Guideline: Here is subject_object list: [person,pillow,bag,shows,cup,bottle,book,table,chair,hands,screen]. Subjects and object at edge of the tuple should be exact the same as ones in the subject_object list.
    Thinking 4: Guideline: all relations should make sense, and person is more important.
    Thinking 5: Guideline: find the at most ten important tuples based on above thinkings and generate the final Answer:
    [["person", "sitting_on", "bed"],
    ["person", "near", "pillow"],
    ["person", "holding", "bag"],
    ["person", "wearing", "shoe"],
    ["cup", "on", "table"],
    ["bottle", "near", "cup"],
    ["book", "on", "table"],
    ["screen", "on", "table"],
    ["chair", "in_front_of", "table"]]
    END

    Now based on the example and requirements, generate the same steps of thinkings as the example by repeating the guidelines and then give reasoning process. Do not over thinking too much. Then generate answer and finish the output with END.

    subject_object list: [{}]
    Input text:{}
    ASSISTANT: '''.format(rel_list, rel_list, objstr, desc)
    
    resdict = {
        'video_id': video_id,
        'frame_id': frame_id,
    }
    
    prompt_list.append(prompt)
    res_dict_list.append(resdict)

failed_encode = 0
for idx, response in tqdm(
    enumerate(
        client.text.generation.create(
            model_id="meta-llama/llama-2-70b-chat",
            inputs=prompt_list,
            parameters=TextGenerationParameters(
                max_new_tokens=2048,
                min_new_tokens=20,
                return_options=TextGenerationReturnOptions(
                    input_text=True,
                ),
            ),
        )
    ),
    total=len(prompt_list)):

    result = response.results[0]
    res_dict = res_dict_list[idx]
    res_dict['input'] = result.input_text
    res_dict['res'] = result.generated_text

    em = None
    rels = []
    try:
        answer = result.generated_text

        startp = [m.start() for m in re.finditer("\[\[", answer)]
        endp = [m.start() for m in re.finditer("]]", answer)]

        add_head = False
        add_tail = False
        
        if len(startp) > 0:
            struct_start = startp[0]
        else:
            struct_start = [m.start() for m in re.finditer('\["', answer)][0]
            add_head = True
            
        if len(endp) > 0:
            struct_end = endp[-1]
        else:
            struct_end = [m.start() for m in re.finditer('"]', answer)][-1]
            add_tail = True

        strlist = answer[struct_start: struct_end + 2]
        if add_head:
            strlist = "[" + strlist
        if add_tail:
            strlist = strlist + "]"
            
        rels = json.loads(strlist)
    except Exception as e:
        pass

    if len(rels) == 0:
        try:
            answer_index = [m.start() for m in re.finditer("Answer:", result.generated_text)][-1]
            answer = result.generated_text[answer_index:]

            startp = [m.start() for m in re.finditer("\[\[", answer)]
            endp = [m.start() for m in re.finditer("]]", answer)]

            add_head = False
            add_tail = False
            
            if len(startp) > 0:
                struct_start = startp[0]
            else:
                struct_start = [m.start() for m in re.finditer('\["', answer)][0]
                add_head = True
                
            if len(endp) > 0:
                struct_end = endp[-1]
            else:
                struct_end = [m.start() for m in re.finditer('"]', answer)][-1]
                add_tail = True

            strlist = answer[struct_start: struct_end + 2]
            if add_head:
                strlist = "[" + strlist
            if add_tail:
                strlist = strlist + "]"
                
            rels = json.loads(strlist)
        except Exception as e:
            em = e
            rels = []
            failed_encode += 1

    resdict['rels'] = rels
    resf.write(json.dumps(resdict) + '\n')
    resf.flush()

    if len(rels) == 0:
        print(findex)
        print(result.generated_text)
        print(em)
        failed_encode += 1

print('failed_encode', failed_encode)
resf.close()
