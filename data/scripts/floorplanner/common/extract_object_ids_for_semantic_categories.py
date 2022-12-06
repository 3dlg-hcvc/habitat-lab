import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

goal_categories_file_path = (
    "data/scene_datasets/floorplanner/v1/goal_categories.yaml"
)
model_stats_path = (
    "data/scene_datasets/floorplanner/v1/stats/fp_model_stats.tsv"
)
semantics_out_path = "data/scene_datasets/floorplanner/v1/configs/semantics"
os.makedirs(semantics_out_path, exist_ok=True)

with open(goal_categories_file_path, "r") as f:
    goal_categories = sorted(yaml.safe_load(f))

data = pd.read_csv(model_stats_path, sep="\t")

for cat in tqdm(goal_categories):
    cat_obj_ids = data.loc[data["category"] == cat]["id"].to_list()
    with open(os.path.join(semantics_out_path, f"{cat}.yaml"), "w") as f:
        yaml.dump(cat_obj_ids, f)
