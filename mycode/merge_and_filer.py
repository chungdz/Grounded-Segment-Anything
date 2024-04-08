import pandas as pd
import json
from tqdm import trange, tqdm
import math

all_lines = []
for i in range(16):
    with open("llava_result/llava_" + str(i) + '.txt') as f:
        curlines = f.readlines()
        print(len(curlines))
        all_lines += curlines

print(len(all_lines))

target = pd.read_csv("Video_Keyframe_IDs.csv")

target_set = set()
for index, row in tqdm(target.iterrows(), total=len(target)):
    video_id = row["video_id"]
    frame_ids = json.loads(row["Keyframe_IDs"].replace("'", '"'))
    for frame_id in frame_ids:
        target_set.add((video_id, frame_id))

newlines = []
for line in all_lines:

    line_dict = json.loads(line)
    video_id = line_dict['video_id']
    frame_id = line_dict['frame_id']
    desc = line_dict['desc']
    objstr = line_dict['objstr']

    if (video_id, frame_id) in target_set:
        newlines.append(line)


perfile = math.ceil(len(newlines) / 2)
print(len(newlines), perfile)
for j in range(2):
    new_file = open("llava_result/filtered/" + str(j) + '.txt', "w")
    new_file.writelines(newlines[j * perfile: (j + 1) * perfile])
    new_file.close()

new_file = open("llava_result/llava_filtered.txt", "w")
new_file.writelines(newlines)

# ids found not in Video_Keyframe_IDs.csv
missing_fid = json.load(open("missing_fid.json", 'r'))
missing_fid = set([(x[0], x[1]) for x in missing_fid])

newlines = []
for line in all_lines:

    line_dict = json.loads(line)
    video_id = line_dict['video_id']
    frame_id = line_dict['frame_id']
    desc = line_dict['desc']
    objstr = line_dict['objstr']

    if (video_id, frame_id) in missing_fid:
        newlines.append(line)

new_file = open("llava_result/filtered/2.txt", "w")
new_file.writelines(newlines)
