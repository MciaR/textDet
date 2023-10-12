import shutil
import os
import json
import random
import tqdm
import pandas as pd
from pandas import DataFrame as df

DATA_PATH = "data/trainset"
VAL_RATIO = 0.1

image_path = os.path.join(DATA_PATH, "train")
ocr_anno_path = os.path.join(DATA_PATH, "train_ocr_results")
ocr_anno_files = os.listdir(ocr_anno_path)

coco_anno_path = os.path.join(DATA_PATH, "train.json")

with open(coco_anno_path, 'r') as f:
    coco_anno = json.load(f)

train_num = len(coco_anno["images"])
val_num = int(train_num * VAL_RATIO)
val_set = set()

while len(val_set) < val_num:
    rand_num = random.randint(0, train_num)
    val_set.add(rand_num)

val_file = {}
val_anno = []

for idx in tqdm(val_set):
    img = coco_anno["images"][idx]
    img_name = img["file_name"]
    img_path = os.path.join(image_path, img_name)
    img_id = img["id"]


print(coco_anno)