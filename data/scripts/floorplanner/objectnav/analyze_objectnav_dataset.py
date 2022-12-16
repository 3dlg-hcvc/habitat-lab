import csv
import gzip
import json
import os
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm

failure_cases_dir = (
    "data/datasets/objectnav/floorplanner/v0.1.0/viz/failure_cases"
)
train_episodes_dir = (
    "data/datasets/objectnav/floorplanner/v0.1.0/train/content"
)
stats_out_dir = "data/datasets/objectnav/floorplanner/v0.1.0/stats"
stats_out_plot_dir = "data/datasets/objectnav/floorplanner/v0.1.0/stats/plots"
goal_categories_path = (
    "data/scene_datasets/floorplanner/v1/goal_categories_6.yaml"
)

with open(goal_categories_path, "r") as f:
    goal_categories = yaml.safe_load(f)

os.makedirs(stats_out_dir, exist_ok=True)
os.makedirs(stats_out_plot_dir, exist_ok=True)

num_goals_dict = {}  # number of goal instances created for each category
num_episodes_dict = {}  # number of episodes created for each category
num_scenes_dict = {}  # number of scenes that have episodes for each category
num_failed_goals_dict = (
    {}
)  # number of goal instances NOT created for each category

for goal_cat in goal_categories:
    num_goals_dict[goal_cat] = 0
    num_episodes_dict[goal_cat] = 0
    num_scenes_dict[goal_cat] = 0
    num_failed_goals_dict[goal_cat] = 0

scenes = os.listdir(train_episodes_dir)

for scene in tqdm(scenes):

    # failure cases
    scene_failure_cases_path = os.path.join(
        failure_cases_dir, scene.split(".")[0]
    )
    if not os.path.exists(scene_failure_cases_path):
        # print(f"No failures in scene: {scene}")
        continue
    for scene_case in os.listdir(scene_failure_cases_path):
        cat = "_".join([x for x in scene_case.split("_") if x.isalpha()])
        num_failed_goals_dict[cat] += 1

    # success cases
    train_episode_path = os.path.join(train_episodes_dir, scene)
    if os.path.exists(train_episode_path):
        with gzip.open(train_episode_path, "r") as fin:
            train_episodes = json.loads(fin.read().decode("utf-8"))

    for goal_key, goal in train_episodes["goals_by_category"].items():

        if len(goal["goals"]):
            cat = goal["goals"][0]["object_category"]

        num_scenes_dict[cat] += 1
        num_goals_dict[cat] += len(goal["goals"])

    for ep in train_episodes["episodes"]:
        ep_cat = ep["object_category"]
        num_episodes_dict[ep_cat] += 1

# plotting and writing to csv
for stats_name, stats in zip(
    ["num_episodes_dict", "num_goals_dict", "num_scenes_dict"],
    [num_episodes_dict, num_goals_dict, num_scenes_dict],
):

    plt.figure(figsize=(24, 8))
    sns.barplot(x=[x[:9] for x in stats.keys()], y=[x for x in stats.values()])
    plt.xticks(rotation=30)
    fig_path = os.path.join(stats_out_plot_dir, f"{stats_name}.jpg")
    plt.savefig(fig_path, dpi=300)

    sorted_stats = {
        k: v for k, v in sorted(stats.items(), key=lambda item: item[1])
    }

    plt.figure(figsize=(24, 8))
    sns.barplot(
        x=[x[:9] for x in sorted_stats.keys()],
        y=[x for x in sorted_stats.values()],
    )
    plt.xticks(rotation=30)
    fig_path = os.path.join(stats_out_plot_dir, f"{stats_name}_sorted.jpg")
    plt.savefig(fig_path, dpi=300)

    with open(os.path.join(stats_out_dir, f"{stats_name}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(stats.items())


for stats_name, stats in zip(
    ["num_failed_goals_dict"], [num_failed_goals_dict]
):
    df1 = pd.DataFrame.from_dict(
        num_failed_goals_dict, orient="index", columns=["failed_goals"]
    )
    df1["category"] = [x[:9] for x in num_failed_goals_dict.keys()]
    df1["successfully_created_goals"] = num_goals_dict.values()
    df1.plot(
        kind="bar",
        stacked=True,
        color=["lime", "green"],
        figsize=(24, 10),
        rot=40,
        ylabel="number of goals",
    )
    fig_path = os.path.join(stats_out_plot_dir, f"{stats_name}_vs_success.jpg")
    plt.savefig(fig_path, dpi=300)
