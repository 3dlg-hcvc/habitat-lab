import os
import random

import yaml


def generate_fp_dataset_splits(scenes_dir_path):
    scenes = sorted(os.listdir(scenes_dir_path))
    scenes = [x.split(".")[0] for x in scenes if x.endswith(".json")]
    splits = ['train', 'val', 'test']
    scene_splits_dict = {"train": [], "test": [], "val": []}
    for scene in scenes:
        for split in splits:
            if split.capitalize() in scene:
                scene_splits_dict[split].append(scene)

    with open(
        os.path.join(splits_yaml_output_path, "procthor_scene_splits.yaml"), "w+"
    ) as f:
        yaml.dump(scene_splits_dict, f, sort_keys=False)

if __name__ == "__main__":
    scenes_dir_path = "data/scene_datasets/ai2thor-hab/v0.0.9/configs/scenes/ProcTHOR"
    splits_yaml_output_path = "data/scene_datasets/ai2thor-hab/v0.0.9"
    generate_fp_dataset_splits(scenes_dir_path)
