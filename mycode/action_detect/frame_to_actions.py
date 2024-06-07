import json

# load the action classes.
def load_action_classes(action_classes_path = 'action_classes.txt'):
    actions_dict = {}
    with open(action_classes_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                action_id, action_description = parts
                actions_dict[action_description] = action_id
    return actions_dict

# load the query database
def load_prediction_result(prediction_result_path = 'masked_result.json'):
    with open(prediction_result_path, 'r') as file:
        result_dict = json.load(file)
    return result_dict

def frame_to_time(frame_number, fps=24):
    return frame_number / fps

# API to query
def frame_to_actions(video_id, frame_number, actions_dict, result_dict):
    result_actions = []
    sec = frame_to_time(int(frame_number.lstrip('0')))
    for prediction in result_dict[video_id]:
        if sec >= prediction["segment"][0] and sec <= prediction["segment"][1] and prediction["label"] in actions_dict:
            result_actions.append((actions_dict[prediction["label"]], prediction["score"]))
    return result_actions

