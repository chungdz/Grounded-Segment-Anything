import argparse
import os
import copy
import json
import sys
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dict_root_path', type=str, default='/home/azon/data/star/', help='path all images')
parser.add_argument('--output_path', type=str, default='./mapping.json', help='path to save annotations')
args = parser.parse_args()

vocab = {}
vlist = ['action_classes.txt', 'object_classes.txt', 'relationship_classes.txt', 'verb_classes.txt']
for vname in vlist:
    with open(os.path.join(args.dict_root_path, 'classes/{}'.format(vname)), 'r') as f:
        for line in f.readlines():
            arr = line.split()
            vocab[arr[0]] = ' '.join(arr[1:])

newd = defaultdict(dict)
dlist = ['train', 'val', 'test']
for dname in dlist:
    annotations = json.load(open(os.path.join(args.dict_root_path, 'Question_Answer_SituationGraph/STAR_{}.json'.format(dname)), 'r'))
    for ann in annotations:
        vkey = ann['video_id']
        for frame_id, graph in ann['situations'].items():
            bbox = graph['bbox']
            bbox_labels = graph['bbox_labels']

            rel_pairs = []
            for r1, r2 in graph['rel_pairs']:
                rel_pairs.append([vocab[r1], vocab[r2]])

            rel_labels = []
            for r in graph['rel_labels']:
                rel_labels.append(vocab[r])

            actions = []
            for a in graph['actions']:
                actions.append(vocab[a])
            
            newd[vkey][frame_id] = {
                'bbox': bbox,
                'bbox_labels': bbox_labels,
                'rel_pairs': rel_pairs,
                'rel_labels': rel_labels,
                'actions': actions
            }

json.dump(newd, open(args.output_path, 'w'))

