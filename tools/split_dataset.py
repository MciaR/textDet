import shutil
import os
import json
import random
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame as df

DATA_PATH = "data/trainvalset"
VAL_RATIO = 0.1
TARGET_VAL_PATH = "data/valset"
TARGET_TRAIN_PATH = "data/trainset"

image_path = os.path.join(DATA_PATH, "train")
ocr_anno_path = os.path.join(DATA_PATH, "train_ocr_results")

coco_anno_path = os.path.join(DATA_PATH, "train.json")

with open(coco_anno_path, 'r') as f:
    coco_anno = json.load(f)

train_num = len(coco_anno["images"])
val_num = int(train_num * VAL_RATIO)
val_set = set()

while len(val_set) < val_num:
    rand_num = random.randint(0, train_num)
    val_set.add(rand_num)

pd_train_img = pd.DataFrame(coco_anno["images"])
pd_train_anno = pd.DataFrame(coco_anno["annotations"])
pd_train_cate = pd.DataFrame(coco_anno["categories"])

# ================== get val dataset =================
val_file = {}
val_anno = []
val_images = []
val_categories = coco_anno["categories"]

if not os.path.exists(os.path.join(TARGET_VAL_PATH, "val")):
    os.makedirs(os.path.join(TARGET_VAL_PATH, "val"))

if not os.path.exists(os.path.join(TARGET_VAL_PATH, "val_ocr_results")):
    os.makedirs(os.path.join(TARGET_VAL_PATH, "val_ocr_results"))

for idx in tqdm(val_set):
    img = coco_anno["images"][idx]
    img_name = img["file_name"]
    img_path = os.path.join(image_path, img_name)
    img_id = img["id"]

    pd_anno = pd_train_anno[pd_train_anno.image_id==img_id]
    anno_json = json.loads(pd_anno.to_json(orient='records'))
    val_anno.extend(anno_json)
    val_images.append(img)

    img_target_path = os.path.join(TARGET_VAL_PATH, "val", img_name)
    shutil.copyfile(img_path, img_target_path)

    img_prefix = img_name.split('.jpg')[0]

    ocr_path = os.path.join(ocr_anno_path, img_prefix + ".json")
    ocr_target_path = os.path.join(TARGET_VAL_PATH, "val_ocr_results", img_prefix + ".json")
    shutil.copyfile(ocr_path, ocr_target_path)

val_file["annotations"] = val_anno
val_file["images"] = val_images
val_file["categories"] = val_categories

with open(os.path.join(TARGET_VAL_PATH, "val.json"), "w") as f:
    f.write(json.dumps(val_file))

if os.path.exists(os.path.join(TARGET_VAL_PATH, "val")) and os.path.exists(os.path.join(TARGET_VAL_PATH, "val.json")):
    print("Create Val Dataset Successfully!")
    print(f"val images len: {len(val_images)}\n val annos count: {len(val_anno)}")

# ================== get train dataset =================
if not os.path.exists(os.path.join(TARGET_TRAIN_PATH, "train")):
    os.makedirs(os.path.join(TARGET_TRAIN_PATH, "train"))

if not os.path.exists(os.path.join(TARGET_TRAIN_PATH, "train_ocr_results")):
    os.makedirs(os.path.join(TARGET_TRAIN_PATH, "train_ocr_results"))

train_file = {}
train_anno = []
train_images = []
train_categories = coco_anno["categories"]

for idx, img in tqdm(enumerate(coco_anno["images"])):
    if idx not in val_set:
        img_name = img["file_name"]
        img_path = os.path.join(image_path, img_name)
        img_id = img["id"]

        pd_anno = pd_train_anno[pd_train_anno.image_id==img_id]
        anno_json = json.loads(pd_anno.to_json(orient='records'))
        train_anno.extend(anno_json)
        train_images.append(img)

        img_target_path = os.path.join(TARGET_TRAIN_PATH, "train", img_name)
        shutil.copyfile(img_path, img_target_path)

        img_prefix = img_name.split('.jpg')[0]

        ocr_path = os.path.join(ocr_anno_path, img_prefix + ".json")
        ocr_target_path = os.path.join(TARGET_TRAIN_PATH, "train_ocr_results", img_prefix + ".json")
        shutil.copyfile(ocr_path, ocr_target_path)

train_file["annotations"] = train_anno
train_file["images"] = train_images
train_file["categories"] = train_categories

with open(os.path.join(TARGET_TRAIN_PATH, "train.json"), "w") as f:
    f.write(json.dumps(val_file))

if os.path.exists(os.path.join(TARGET_TRAIN_PATH, "train")) and os.path.exists(os.path.join(TARGET_TRAIN_PATH, "train.json")):
    print("Create Val Dataset Successfully!")
    print(f"train images len: {len(train_images)}\n train annos count: {len(train_anno)}")