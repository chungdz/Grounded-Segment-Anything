import json
from tqdm import trange, tqdm
from collections import defaultdict
import pandas as pd
from frame_to_actions import load_action_classes, load_prediction_result, frame_to_actions
import numpy as np

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

actions_dict = load_action_classes()
result_dict = load_prediction_result()

all_scores = []
all_gt_scores = []
all_gt_index = []
for i, file in enumerate(all_files):
    with open(graph_path + file, 'r') as f:
        data = json.load(f)
        
        for q in tqdm(data):
            qid = q['question_id']
            video_id = q2v[qid]
            fdict = q['situations']
            newfdict = defaultdict(dict)
            for fid, finfo in fdict.items():
                if fid in video_dict[video_id]:
                    actions = frame_to_actions(video_id, fid, actions_dict, result_dict)[:8]
                    if len(actions) == 0:
                        continue
                    pred_actions = [x[0] for x in actions]
                    pred_scores = [x[1] for x in actions]
                    gt_actions = finfo['actions']
                    all_scores.extend(pred_scores)
                    gt_scores = []
                    for action in gt_actions:
                        try:
                            aindex = pred_actions.index(action)
                            all_gt_index.append(aindex)
                            gt_scores.append(pred_scores[aindex])
                        except:
                            pass
                    all_gt_scores.extend(gt_scores)

inters = np.arange(0, 105, 5)
print(np.percentile(all_scores, inters))
#[0.0012 0.1177 0.132  0.1433 0.1524 0.1608 0.1683 0.1756 0.1827 0.1904 0.1985 0.2074 0.2173 0.2291 0.2419 0.2577 0.2747 0.2982 0.328 0.3734 0.6446]

print(np.percentile(all_gt_scores, inters))
# [0.0109 0.1374 0.1583 0.1737 0.1832 0.1946 0.206  0.2149 0.2248 0.234 0.245  0.2596 0.2742 0.2879 0.3055 0.3192 0.3342 0.3553 0.3819 0.4288 0.6446]
print(np.mean(all_gt_scores))

print(np.percentile(all_gt_index, inters))
# [0. 0. 0. 0. 0. 0. 1. 1. 1. 2. 2. 2. 3. 3. 3. 4. 4. 5. 6. 7. 7.]


