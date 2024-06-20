import json
from tqdm import trange, tqdm
from collections import defaultdict
import pandas as pd
from action_detect.frame_to_actions import load_action_classes, load_prediction_result, frame_to_actions
from action_detect.same_class import clist

asame = {}
for aset in clist:
    for a in aset:
        asame[a] = aset

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

graph_path = "/home/azon/data/star/Question_Answer_SituationGraph/"
all_files = ["Feasibility_test.json", 
             "Interaction_test.json",
                "Prediction_test.json",
                "Sequence_test.json"
            ]
output_path = "/home/azon/data/star/new_graphs2/"
output_files = ["Feasibility.json", "Interaction.json", "Prediction.json", "Sequence.json"]

actions_dict = load_action_classes('action_detect/action_classes.txt')
result_dict = load_prediction_result('action_detect/masked_result.json')

missing_fid = []
zero_actions = []
threshold = 0
for i, file in enumerate(all_files):
    with open(graph_path + file, 'r') as f:
        data = json.load(f)
        newdata = []
        
        for q in tqdm(data):
            qid = q['question_id']
            video_id = q2v[qid]
            fdict = q['situations']
            newfdict = defaultdict(dict)
            for fid, finfo in fdict.items():
                if fid in video_dict[video_id]:
                    actions = frame_to_actions(video_id, fid, actions_dict, result_dict)[:8]
                    if len(actions) == 0:
                        # print("0 actions", video_id, fid, newfdict[fid]['actions'])
                        zero_actions.append([video_id, fid])
                        # newfdict[fid]['actions'] = ["p000"]
                        continue
                    newfdict[fid]['rel_labels'] = video_dict[video_id][fid]['rel_labels']
                    newfdict[fid]['rel_pairs'] = video_dict[video_id][fid]['rel_pairs']
                    newfdict[fid]['bbox_labels'] = video_dict[video_id][fid]['bbox_labels']
                    newfdict[fid]['actions'] = []
                    notadd = set()
                    for action, score in actions:
                        if score >= threshold or len(newfdict[fid]['actions']) == 0:
                            if action not in notadd:
                                newfdict[fid]['actions'].append(action)
                                # notadd.update(asame[action])
                else:
                    # print("Not found", fid, video_id)
                    missing_fid.append([video_id, fid])

            newdata.append([qid, newfdict])

        with open(output_path + output_files[i], 'w') as f:
            json.dump(newdata, f)

print(len(missing_fid))
print(len(zero_actions))

json.dump(missing_fid, open("/home/azon/data/star/new_graphs2/missing_fid.json", 'w'))
json.dump(zero_actions, open("/home/azon/data/star/new_graphs2/zero_actions.json", 'w'))


