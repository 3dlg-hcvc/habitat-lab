import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
import yaml
from tqdm import tqdm

goal_categories_file_path = (
    "data/scene_datasets/floorplanner/v1/goal_categories.yaml"
)
model_stats_path = (
    "data/scene_datasets/floorplanner/v1/stats/fp_model_stats.tsv"
)
object_dataset_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/scene_datasets/floorplanner/with-objects/config/configs/furniture-noq"
stats_out_path = "data/scene_datasets/floorplanner/v1/stats/"
objects_in_scenes_stats_path = "/nethome/mkhanna37/flash1/sbd/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/stats/objects.json"
scene_splits_path = "/nethome/mkhanna37/flash1/sbd/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/scene_splits.yaml"

with open(goal_categories_file_path, "r") as f:
    goal_categories = sorted(yaml.safe_load(f))

with open(scene_splits_path, "r") as f:
    scene_splits = yaml.safe_load(f)

with open(objects_in_scenes_stats_path, "r") as f:
    objects_in_scenes_stats = json.load(f)

model_stats = pd.read_csv(model_stats_path, sep="\t")
obj_ids_per_split_out_path = os.path.join(
    stats_out_path, "object_ids_per_split.json"
)
obj_ids_per_split = {}

for split in ["test", "val", "train"]:
    cat_size_dict = {}
    cat_size_means = {}
    obj_ids_per_split[split] = {}

    for cat in tqdm(goal_categories, desc=split):
        cat_size_dict[cat] = {
            "area": [],
            "volume": [],
            "scale": [],
            "count": 0,
        }
        cat_obj_ids = model_stats.loc[model_stats["category"] == cat][
            "id"
        ].to_list()
        if len(cat_obj_ids) == 0:
            print(f"Warning: No object IDs found for: {cat}")
            goal_categories.remove(cat)
            continue

        split_cat_obj_ids = []

        for obj_id in cat_obj_ids:
            for scene in objects_in_scenes_stats[obj_id][
                "scene_counts"
            ].keys():
                if scene in scene_splits[split]:
                    # print(f"{obj_id} belongs in {split} split.")
                    split_cat_obj_ids.append(obj_id)
                    break
            # print(f"{obj_id} does not belong in {split} split.")

        obj_ids_per_split[split][cat] = split_cat_obj_ids

        for obj_id in split_cat_obj_ids:
            obj_mesh_path = os.path.join(object_dataset_path, obj_id + ".glb")
            mesh = trimesh.load(obj_mesh_path)
            cat_size_dict[cat]["volume"].append(mesh.bounding_box.volume)
            cat_size_dict[cat]["scale"].append(mesh.bounding_box.scale)
            cat_size_dict[cat]["area"].append(mesh.area)
        cat_size_means[cat] = np.array(cat_size_dict[cat]["scale"]).mean()
        cat_size_dict[cat]["count"] = len(split_cat_obj_ids)

    data = [cat_size_dict[x]["scale"] for x in goal_categories]
    plt.figure(figsize=(100, 30), facecolor="w")
    labels = [x[:6] for x in goal_categories]
    boxplot = plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        showfliers=False,
    )
    plt.tick_params(axis="both", which="major", labelsize=35)

    for element in boxplot.keys():
        if element == "means":
            plt.setp(boxplot[element], color="red", linewidth=4)
        else:
            plt.setp(boxplot[element], color="blue", linewidth=3)

    for patch in boxplot["boxes"]:
        patch.set(facecolor="cyan")

    plt.title(
        "Floorplanner object dataset (3D diagonal) size distribution",
        fontsize=50,
    )
    plt.savefig(
        os.path.join(stats_out_path, f"size_distribution_plot_{split}.png")
    )

    csv_file = f"category_size_stats_{split}.csv"
    os.makedirs(stats_out_path, exist_ok=True)
    with open(os.path.join(stats_out_path, csv_file), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(cat_size_means.items())

    with open(obj_ids_per_split_out_path, "w") as f:
        json.dump(obj_ids_per_split, f, indent=4)
