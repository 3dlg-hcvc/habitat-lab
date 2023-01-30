import os
import gzip
import json
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

from collections import defaultdict
import random

import torch
from dgl.geometry import farthest_point_sampler

from habitat.core.utils import DatasetJSONEncoder

NUM_EPISODES = 10000
CATEGORY_PATH = 'data/scene_datasets/floorplanner/goals/33_goal_categories.yaml'
SCENE_DSET_STATS_PATH = 'data/scene_datasets/fp_v1_with_objects_stats.tsv'
SCENES_PATH = 'data/datasets/objectnav/floorplanner/v0.1.1/val/content/'
OUT_PATH = 'data/datasets/objectnav/floorplanner/v0.1.1/val/content_sampled/'

def seed(seed=0):
    random.seed(seed)

def sample_episode_per_cat_random(cat_episodes, num_episodes):
    return random.sample(cat_episodes, num_episodes)

def percentage_split(seq, percentages):
    cdf = np.cumsum(percentages).astype(np.float16)
    stops = list(map(int, cdf * len(seq)))
    return [len(seq[a:b]) for a, b in zip([0]+stops, stops)]

def _get_num_eps_per_scene(scene_info, goal_cat, num_eps):

    cat_scene_weights = {}

    for scene in scene_info:
        scene_id = scene["scene"]
        if goal_cat in scene["inst_counts"].keys():
            cat_scene_weights[scene_id] = scene["nav_area"]
        else:
            cat_scene_weights[scene_id] = 0
    
    # normalize weights
    total_sum = sum([x for x in cat_scene_weights.values() if x!=0])
    # buffer_record = {}
    percentages = []
    for (scene, weight) in cat_scene_weights.items():
        # num_episodes_reqd = int(weight * NUM_EPISODES_PER_CAT / total_sum)
        num_episodes_reqd = weight / total_sum
        percentages.append(num_episodes_reqd)

    seq = np.arange(num_eps).tolist()
    lengths = percentage_split(seq, percentages)

    for i, (scene, weight) in enumerate(cat_scene_weights.items()):
        cat_scene_weights[scene] = lengths[i]

    return cat_scene_weights

def fps_wrapper(start_points, num_points):

    points = torch.Tensor(start_points).unsqueeze(0)
    idxes = farthest_point_sampler(points, npoints=num_points)

    return idxes[0].cpu().numpy()

def sample_cat_episode_per_scene(scene_data, scene_info, category, num_eps):

    # get number of eps per scene by nav area weighting

    num_eps_per_scene = _get_num_eps_per_scene(scene_info, category, num_eps=num_eps)

    # sample eps by fps
    cat_eps_per_scene = {}
    for scene_id, num_eps in num_eps_per_scene.items():
        
        if num_eps == 0:
            continue
        cat_eps = [x for x in scene_data[scene_id]['episodes'] if x['object_category'] == category]

        eps_start_positions = [x['start_position'] for x in cat_eps]
        fp_start_idxes = fps_wrapper(eps_start_positions, num_eps)

        if len(fp_start_idxes) == 0:
            import pdb
            pdb.set_trace()
        cat_eps = [cat_eps[x] for x in fp_start_idxes]

        cat_eps_per_scene[scene_id] = cat_eps

    return cat_eps_per_scene

def main():

    seed()
    
    with open(CATEGORY_PATH, 'r') as fp:
        CATEGORIES = yaml.safe_load(fp)

    scenes = os.listdir(SCENES_PATH)
    scene_data = {}

    for sc in tqdm(scenes, desc='loading scene episodes'):
        with gzip.open(os.path.join(SCENES_PATH, sc), "r") as fin:
            ep_data = json.loads(fin.read().decode("utf-8"))

        scene_data[sc.replace('.json.gz', '')] = ep_data

    # scene info
    scene_dataset_stats = pd.read_csv(SCENE_DSET_STATS_PATH, sep='\t', index_col='scene')

    scene_info = []
    for scene_id in tqdm(scene_data):
        scene_info_obj = {"scene": scene_id, "inst_counts": {}, "num_episodes": {}}
        nav_area = scene_dataset_stats.loc[scene_id]["navigable_area"]
        scene_info_obj["nav_area"] = nav_area
        
        goals_by_category = scene_data[scene_id]["goals_by_category"]

        for goal in goals_by_category:
            goal_id = "_".join([x for x in goal.split("_") if x.isalpha()])
            scene_info_obj["inst_counts"][goal_id] = len(scene_data[scene_id]["goals_by_category"][goal]["goals"])
            scene_info_obj["num_episodes"][goal_id] = len([x for x in scene_data[scene_id]["episodes"] if x['object_category'] == goal_id])

        scene_info.append(scene_info_obj)

    # get number of episodes per category
    eps_per_cat = {cat: int(NUM_EPISODES / len(CATEGORIES)) for cat in CATEGORIES}
    remainder = NUM_EPISODES % len(CATEGORIES)
    for _ in range(remainder):
        eps_per_cat[random.choice(CATEGORIES)] += 1

    scene_data_val = defaultdict(dict)
    for scene_id in scene_data.keys():
        scene_data_val[scene_id]['episodes'] = []

    for cat, num_eps in tqdm(eps_per_cat.items(), desc='per category sampling'):
        cat_episodes_per_scene = sample_cat_episode_per_scene(scene_data, scene_info, cat, num_eps) 

        for scene_id, eps in cat_episodes_per_scene.items():
            scene_data_val[scene_id]['episodes'].extend(eps)

    for scene_id in scene_data.keys():
        scene_data[scene_id]['episodes'] = scene_data_val[scene_id]['episodes']

    os.makedirs(OUT_PATH, exist_ok=True)
    for sc, ep_data in scene_data.items():
        print(sc, ep_data['episodes'].__len__(), ep_data.keys())
        with gzip.open(os.path.join(OUT_PATH, sc + '.json.gz'), "wt") as fp:
            fp.write(DatasetJSONEncoder().encode(ep_data))

if __name__ == '__main__':
    main()