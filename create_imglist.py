import json
import os

with open(
        './data/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json',
        'r') as f:
    train_list = json.load(f)

with open('./train_list.lst', 'w') as fout:
    for i, item in enumerate(train_list):
        line = '{}\t{}\t{}\n'.format(i, item['label_id'], item['image_id'])
        fout.write(line)

with open(
        './data/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',
        'r') as f:
    valid_list = json.load(f)

with open('./valid_list.lst', 'w') as fout:
    for i, item in enumerate(valid_list):
        line = '{}\t{}\t{}\n'.format(i, item['label_id'], item['image_id'])
        fout.write(line)
