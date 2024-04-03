import json
from tqdm import trange, tqdm
from collections import defaultdict
import pandas as pd

label_path = "/home/azon/data/star/classes/"

obj_path = label_path + "object_classes.txt"

obj_dict = {"shoes": "o012"}
with open(obj_path) as f:
    lines = f.readlines()
    for line in lines:
        mapping = line.strip('\n')
        index, words = mapping.split(' ')
        word_list = words.split('/')
        for w in word_list:
            obj_dict[w] = index

relation_path = label_path + "relationship_classes.txt"
rel_dict = {}
with open(relation_path) as f:
    lines = f.readlines()
    for line in lines:
        mapping = line.strip('\n')
        index, w = mapping.split(' ')
        rel_dict[w] = index

llama_path = "/home/azon/data/star/llama_result/"
files = ["llama_0.txt", "llama_11.txt", "llama_10.txt"]
video_dict = defaultdict(dict)
for file in files:
    with open(llama_path + file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_dict = json.loads(line)
            video_id = line_dict['video_id']
            frame_id = line_dict['frame_id']
            
            finfo = {
                "rel_labels": [],
                "rel_pairs": [],
                'bbox_labels': []
            }

            all_obj = set()
            relations = line_dict['rels']
            for rtuple in relations:
                if len(rtuple) != 3:
                    continue
                o1, r, o2 = rtuple
                if o1 in obj_dict and o2 in obj_dict and r in rel_dict:
                    finfo["rel_labels"].append(rel_dict[r])
                    finfo["rel_pairs"].append([obj_dict[o1], obj_dict[o2]])
                    all_obj.add(obj_dict[o1])
                    all_obj.add(obj_dict[o2])
            finfo['bbox_labels'] = list(all_obj)
            
            video_dict[video_id][frame_id] = finfo
            
vk = list(video_dict.keys())

target = pd.read_csv("/home/azon/data/star/Video_Keyframe_IDs.csv")

q2v = {}
for index, row in tqdm(target.iterrows(), total=len(target)):
    video_id = row["video_id"]
    question_id = row["question_id"]
    q2v[question_id] = video_id

graph_path = "/home/azon/data/star/rpl_act_8/"
all_files = ["Interaction_GT_Sem/star_Interaction_action_transition_model.json", 
             "Feasibility_GT_Sem/star_Feasibility_action_transition_model.json",
             "Prediction_GT_Sem/star_Prediction_action_transition_model.json",
             "Sequence_GT_Sem/star_Sequence_action_transition_model.json"]
output_path = "/home/azon/data/star/new_graphs/"
output_files = ["Interaction.json", "Feasibility.json", "Prediction.json", "Sequence.json"]

for i, file in enumerate(all_files):
    with open(graph_path + file, 'r') as f:
        data = json.load(f)
        
        for q in tqdm(data):
            qid = q[0]
            video_id = q2v[qid]
            fdict = q[1]
            prev = None
            for fid, finfo in fdict.items():
                if 'padding' in fid:
                    finfo['rel_labels'] = prev['rel_labels']
                    finfo['rel_pairs'] = prev['rel_pairs']
                    finfo['bbox_labels'] = prev['bbox_labels']
                else:
                    finfo['rel_labels'] = video_dict[video_id][fid]['rel_labels']
                    finfo['rel_pairs'] = video_dict[video_id][fid]['rel_pairs']
                    finfo['bbox_labels'] = video_dict[video_id][fid]['bbox_labels']

                prev = finfo

        with open(output_path + output_files[i], 'w') as f:
            json.dump(data, f)
# check relation list

import json

files = ['Interaction_train.json', 'Interaction_val.json', 'Interaction_test.json', 
         'Prediction_train.json', 'Prediction_val.json', 'Prediction_test.json', 
         'Sequence_train.json', 'Sequence_val.json', 'Sequence_test.json', 
         'Feasibility_train.json', 'Feasibility_val.json', 'Feasibility_test.json']
for file in files:
    a = json.load(open(file, 'r'))
    res = set()
    for q in a:
        fdict = q['situations']
        for fid, finfo in fdict.items():
            for rel in finfo['rel_labels']:
                    res.add(rel)
    print(file)
    print(res)


import json

files = ["Interaction_GT_Sem/star_Interaction_action_transition_model.json"]
for file in files:
    a = json.load(open(file, 'r'))
    res = set()
    for q in a:
        fdict = q[1]
        for fid, finfo in fdict.items():
            for rel in finfo['rel_labels']:
                    res.add(rel)
    print(file)
    print(res)

