import json
import os

test_anno = 'data/testset/pred.json'
predict_res = 'work_dirs/coco_detection/test.bbox.json'
output_path = 'work_dirs/pred.json'

with open(test_anno, 'r') as f:
    empty_anno = json.load(f)

with open(predict_res, 'r') as f:
    pred_anno = json.load(f)

empty_anno['annotations'] = pred_anno

with open(output_path, 'w') as f:
    f.write(json.dumps(empty_anno))