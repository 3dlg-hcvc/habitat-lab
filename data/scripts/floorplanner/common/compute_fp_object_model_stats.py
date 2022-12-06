import csv
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

with open(goal_categories_file_path, "r") as f:
    goal_categories = sorted(yaml.safe_load(f))

data = pd.read_csv(model_stats_path, sep="\t")

cat_size_dict = {}
cat_size_means = {}

for cat in tqdm(goal_categories):
    cat_size_dict[cat] = {"area": [], "volume": [], "scale": [], "count": 0}
    cat_obj_ids = data.loc[data["category"] == cat]["id"].to_list()
    if len(cat_obj_ids) == 0:
        print(f"Warning: No object IDs found for: {cat}")
        goal_categories.remove(cat)
        continue
    for obj_id in cat_obj_ids:
        obj_mesh_path = os.path.join(object_dataset_path, obj_id + ".glb")
        mesh = trimesh.load(obj_mesh_path)
        cat_size_dict[cat]["volume"].append(mesh.bounding_box.volume)
        cat_size_dict[cat]["scale"].append(mesh.bounding_box.scale)
        cat_size_dict[cat]["area"].append(mesh.area)
    cat_size_means[cat] = np.array(cat_size_dict[cat]["scale"]).mean()
    cat_size_dict[cat]["count"] = len(cat_obj_ids)

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
    "Floorplanner object dataset (3D diagonal) size distribution", fontsize=50
)
plt.savefig(os.path.join(stats_out_path, "size_distribution_plot.png"))

csv_file = "category_size_stats.csv"
os.makedirs(stats_out_path, exist_ok=True)
with open(os.path.join(stats_out_path, csv_file), "w") as csvfile:
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerows(cat_size_means.items())
