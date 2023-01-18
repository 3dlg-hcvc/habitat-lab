import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

version_id = "v0.2.0"

goal_categories_file_path = (
    f"data/scene_datasets/floorplanner/goals/6_goal_categories.yaml"
)
model_stats_path = f"data/scene_datasets/floorplanner/{version_id}/configs/semantics/object_metadata.csv"
semantics_out_path = (
    f"data/scene_datasets/floorplanner/{version_id}/configs/semantics"
)

with open(goal_categories_file_path, "r") as f:
    goal_categories = sorted(yaml.safe_load(f))

data = pd.read_csv(model_stats_path)

for cat in tqdm(goal_categories):
    cat_obj_ids = data.loc[data["main_category"] == cat]["id"].to_list()
    with open(os.path.join(semantics_out_path, f"{cat}.yaml"), "w") as f:
        yaml.dump(cat_obj_ids, f)
