# textDet
Document Detection for "Daguan" Cup Competition
## Download data
Here is train and test data: [data link](https://challenge.datacastle.cn/v3/cmptDetail.html?id=824)<br/>
**when you decompress .zip file, please rename `data/trainset` folder to `data/trainvalset`**<br/>
Using `tools/split_dataset.py` can split train data to train and val data for training evaluating.

## Install Requirements
Thie repo relies on [MMLab/MMDetection](https://github.com/open-mmlab/mmdetection/tree/main), Please install mmdet first. Installation document： https://mmdetection.readthedocs.io/en/latest/get_started.html

## Train
### Train on a single GPU
```
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```
### Train on multiple GPUs
```
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```
### Examples (single GPU)
```
python ./tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```
