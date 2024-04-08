import json
from tqdm import trange, tqdm
from collections import defaultdict
import pandas as pd

target = pd.read_csv("/home/azon/data/star/Video_Keyframe_IDs.csv")

q2v = {}
for index, row in tqdm(target.iterrows(), total=len(target)):
    video_id = row["video_id"]
    question_id = row["question_id"]
    q2v[question_id] = video_id

gt_path = "/home/azon/data/star/Question_Answer_SituationGraph/"
all_files = ["Feasibility_test.json", 
             "Interaction_test.json",
                "Prediction_test.json",
                "Sequence_test.json"
            ]
video_dict = {}
for i, file in enumerate(all_files):
    with open(gt_path + file, 'r') as f:
        data = json.load(f)
        
        for q in tqdm(data):
            qid = q['question_id']
            video_id = q2v[qid]
            fdict = q['situations']
            prev = None
            video_dict[video_id] = fdict

graph_path = "/home/azon/data/star/rpl_act_8/"
all_files = ["Interaction_GT_Sem/star_Interaction_action_transition_model.json", 
             "Feasibility_GT_Sem/star_Feasibility_action_transition_model.json",
             "Prediction_GT_Sem/star_Prediction_action_transition_model.json",
             "Sequence_GT_Sem/star_Sequence_action_transition_model.json"]
output_path = "/home/azon/data/star/new_graphs3/"
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
                elif fid in video_dict[video_id]:
                    finfo['rel_labels'] = video_dict[video_id][fid]['rel_labels']
                    finfo['rel_pairs'] = video_dict[video_id][fid]['rel_pairs']
                    finfo['bbox_labels'] = video_dict[video_id][fid]['bbox_labels']

                prev = finfo

        with open(output_path + output_files[i], 'w') as f:
            json.dump(data, f)