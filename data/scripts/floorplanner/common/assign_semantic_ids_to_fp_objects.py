import json
import os

import pandas as pd
import yaml
from tqdm import tqdm

version_id = "v0.2.0"
obj_metadata_path = f"data/scene_datasets/floorplanner/{version_id}/configs/semantics/object_metadata.csv"
objects_path = f"data/scene_datasets/floorplanner/{version_id}/configs/objects"
goal_categories_path = (
    "data/scene_datasets/floorplanner/goals/33_goal_categories.yaml"
)
semantic_id_mapping_path = f"data/scene_datasets/floorplanner/{version_id}/configs/object_semantic_id_mapping.json"


if __name__ == "__main__":
    obj_metadata = pd.read_csv(obj_metadata_path)

    with open(semantic_id_mapping_path, "r") as f:
        semantic_id_mapping = json.load(f)

    with open(goal_categories_path, "r") as f:
        goal_categories = yaml.safe_load(f)

    for cat in goal_categories:
        cat_semantic_id = semantic_id_mapping[cat]
        obj_ids = obj_metadata[obj_metadata["main_category"] == cat][
            "id"
        ].tolist()
        assert len(obj_ids) == len(set(obj_ids))  # assert no duplicate ids

        for obj_id in tqdm(obj_ids, desc=f"{cat} ({cat_semantic_id})"):
            obj_json_path = os.path.join(
                objects_path, obj_id + ".object_config.json"
            )
            with open(obj_json_path, "r") as f:
                data = json.load(f)
                assert "semantic_id" not in data.keys()
                data["semantic_id"] = cat_semantic_id

            with open(obj_json_path, "w") as f:
                json.dump(data, f)
