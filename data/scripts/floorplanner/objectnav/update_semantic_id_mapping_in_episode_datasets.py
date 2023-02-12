"""
Note: Currently, only updating scene dataset used for modular experiments.
0. Assumes appropriate mapping specified in semantic_id_mapping.json file
1. Updates category id-mapping in semantic.tsv file.
2. Updates object semantic ID in object config json files.
3. Updates goal object ID in objectnav dataset.
"""

import json
import os
import pandas as pd
from tqdm import tqdm
import gzip

# Updates goal object ID in objectnav dataset.

# HM3D:
# mapping = {"chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5}

# FP:
mapping = {"chair": 0, "bed": 1, "potted_plant": 2, "toilet": 3, "tv": 4, "couch": 5}

# THOR
# mapping = {"Chair": 0, "Bed": 1, "HousePlant": 2, "Toilet": 3, "Television": 4, "Sofa": 5}

# file paths to be changed ->
objnav_episode_dataset = "/nethome/mkhanna37/flash1/sbd-latest/scene-builder-datasets/fphab/habitat-lab/data/datasets/objectnav/floorplanner/v0.2.0_6_cat_indoor_10k_per_scene/train/content"
dataset_split_json_path = "/nethome/mkhanna37/flash1/sbd-latest/scene-builder-datasets/fphab/habitat-lab/data/datasets/objectnav/floorplanner/v0.2.0_6_cat_indoor_10k_per_scene/train/train.json.gz"

with gzip.open(dataset_split_json_path, "r") as fin:
    data = json.loads(fin.read().decode("utf-8"))
    data["category_to_task_category_id"] = mapping
    data["category_to_scene_annotation_category_id"] = mapping

with gzip.open(dataset_split_json_path, "wt", encoding="ascii") as f:
    json.dump(data, f)

for scene in tqdm(
    os.listdir(objnav_episode_dataset), desc=f"updating all episode files"
):
    scene_path = os.path.join(objnav_episode_dataset, scene)
    print(scene_path)
    with gzip.open(scene_path, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
        if (
            data["category_to_task_category_id"] == mapping
            and data["category_to_scene_annotation_category_id"] == mapping
        ):
            continue
        data["category_to_task_category_id"] = mapping
        data["category_to_scene_annotation_category_id"] = mapping

    with gzip.open(scene_path, "wt", encoding="ascii") as f:
        json.dump(data, f)
