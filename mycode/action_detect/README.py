'''
    Use case for the function. Please load the module and functions included.
    The load_action_classes & load_prediction_result shall be execute once for performance.
'''


from frame_to_actions import load_action_classes, load_prediction_result, frame_to_actions

actions_dict = load_action_classes('action_detect/action_classes.txt')
result_dict = load_prediction_result('action_detect/masked_result.json')

video_id = "VZE8E"
frame_number = "000006"
actions = frame_to_actions(video_id, frame_number, actions_dict, result_dict)
print(actions) 