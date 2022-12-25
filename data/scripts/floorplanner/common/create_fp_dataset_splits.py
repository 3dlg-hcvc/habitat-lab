import os
import random

import yaml


def generate_fp_dataset_splits(scenes_dir_path, split_ratios):
    random.seed(0)

    train_split_ratio, test_split_ratio, val_split_ratio = split_ratios

    assert (
        round(train_split_ratio + test_split_ratio + val_split_ratio, 3) == 1.0
    )

    scenes = sorted(os.listdir(scenes_dir_path))
    scenes = [x.split(".")[0] for x in scenes if x.endswith(".json")]

    random.shuffle(scenes)

    train_val_size = int(len(scenes) * (train_split_ratio + val_split_ratio))

    train_val_set = scenes[:train_val_size]
    test_set = scenes[train_val_size:]

    train_size = int(
        len(train_val_set)
        * train_split_ratio
        / (train_split_ratio + val_split_ratio)
    )

    train_set = train_val_set[:train_size]
    val_set = train_val_set[train_size:]

    scene_splits_dict = {"train": train_set, "test": test_set, "val": val_set}

    with open(
        os.path.join(splits_yaml_output_path, "scene_splits.yaml"), "w"
    ) as f:
        yaml.dump(scene_splits_dict, f, sort_keys=False)


if __name__ == "__main__":
    scenes_dir_path = "data/scene_datasets/ai2thor-hab/configs/scenes"
    splits_yaml_output_path = "data/scene_datasets/ai2thor-hab/"
    train_split_ratio, test_split_ratio, val_split_ratio = 0.6, 0.2, 0.2
    generate_fp_dataset_splits(
        scenes_dir_path, (train_split_ratio, test_split_ratio, val_split_ratio)
    )
