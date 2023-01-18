import json
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

model_size_path = "/nethome/mkhanna37/flash1/sbd/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/stats/fp_object_models_stats.tsv"
scene_object_stats_path = "/nethome/mkhanna37/flash1/sbd/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/stats/scene_object_stats.csv"
scene_splits_path = "/nethome/mkhanna37/flash1/sbd/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/scene_splits.yaml"
goal_categories_file_path = (
    "data/scene_datasets/floorplanner/v1/goal_categories.yaml"
)
out_path = os.path.join(
    os.path.dirname(scene_object_stats_path), "object_stats_per_scene.json"
)

with open(goal_categories_file_path, "r") as f:
    goal_categories = sorted(yaml.safe_load(f))

model_sizes = pd.read_csv(model_size_path, sep="\t")
scene_object_stats = pd.read_csv(scene_object_stats_path)
with open(scene_splits_path, "r") as f:
    scene_splits = yaml.safe_load(f)

split_dict = {}
for split, scenes in scene_splits.items():
    scenes_obj_dict = {}
    for scene in tqdm(scenes, desc=split):
        scene_obj_dict = {}
        for cat in goal_categories:
            scene_obj_dict[cat] = {
                "all_instances": [],
                "all_sizes": [],
            }

        scene_objects = (
            scene_object_stats[scene_object_stats["id"] == scene]["modelIds"]
            .to_list()[0]
            .split(",")
        )
        for obj_id in scene_objects:
            try:
                obj_cat = model_sizes[model_sizes["id"] == obj_id][
                    "category"
                ].to_list()[0]
            except Exception as e:
                print(
                    f"{obj_id} found in scene_object_stats that does not belong in model dataset csv."
                )
                continue
            if type(obj_cat) != str:
                continue

            if obj_cat in scene_obj_dict.keys():
                scene_obj_dict[obj_cat]["all_instances"].append(obj_id)
                size = model_sizes[model_sizes["id"] == obj_id][
                    "scale"
                ].to_list()[0]
                scene_obj_dict[obj_cat]["all_sizes"].append(size)

        for cat in scene_obj_dict.keys():
            scene_obj_dict[cat]["mean_size"] = np.array(
                scene_obj_dict[cat]["all_sizes"]
            ).mean()
            scene_obj_dict[cat]["median_size"] = np.median(
                np.array(scene_obj_dict[cat]["all_sizes"])
            )
            scene_obj_dict[cat]["unique_instances"] = list(
                set(scene_obj_dict[cat]["all_instances"])
            )
            scene_obj_dict[cat]["num_unique_instances"] = len(
                scene_obj_dict[cat]["unique_instances"]
            )

        scenes_obj_dict[scene] = scene_obj_dict

    split_dict[split] = scenes_obj_dict


import pdb

pdb.set_trace()
with open(out_path, "w") as f:
    json.dump(split_dict, f)
