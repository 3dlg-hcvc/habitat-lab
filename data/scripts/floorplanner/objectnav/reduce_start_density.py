import os
import gzip
import json
from tqdm import tqdm

import numpy as np
import cv2

import random

import torch
from dgl.geometry import farthest_point_sampler

from habitat.core.utils import DatasetJSONEncoder

from sklearn.cluster import AgglomerativeClustering

from visualize_episode import get_objnav_config, get_simulator, _visualize_goal, get_topdown_map, COLOR_MAP

SCENE_DSET_STATS_PATH = 'data/scene_datasets/fp_v1_with_objects_stats.tsv'
SCENES_PATH = 'data/datasets/objectnav/floorplanner/v0.2.0_6_cat_indoor_only/train/content'
OUT_PATH = 'data/datasets/objectnav/floorplanner/v0.2.0_6_cat_indoor_only/train/content_sampled_fps'
NUM_PER_SCENE = 20000 # num of episodes to sample per cluster
VISUALIZE = True

def seed(seed=0):
    random.seed(seed)

def sample_episodes_per_cat(episodes, cat):
    return [x for x in episodes if x['object_category'] == cat]

def cluster_episodes(episodes):

    start_points = [x['start_position'] for x in episodes]
    start_points = np.array(start_points)

    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        affinity="euclidean",
                        distance_threshold=1.0,
                    ).fit(start_points)
    
    return clustering.labels_, clustering.n_clusters_

def fps_wrapper(start_points, num_points):

    points = torch.Tensor(start_points).unsqueeze(0)
    idxes = farthest_point_sampler(points, npoints=num_points)

    return idxes[0].cpu().numpy()

def sample_episodes_incluster_fps(episodes, cluster_labels, num_eps):

    sampled_episodes = []
    clusters = np.unique(cluster_labels)
    for cl in clusters:

        cl_eps = [x for (i, x) in enumerate(episodes) if cluster_labels[i] == cl]
        if num_eps < len(cl_eps):
            start_points = [x['start_position'] for x in cl_eps]
            start_points = np.array(start_points)
            sampled_idx = fps_wrapper(start_points, num_points=num_eps)

            for idx in sampled_idx:
                sampled_episodes.append(cl_eps[idx])
        else:
            sampled_episodes.extend(cl_eps)

    return sampled_episodes

def sample_episodes_with_fps(episodes, num_eps):

    start_points = [x['start_position'] for x in episodes]
    start_points = np.array(start_points)
    sampled_idx = fps_wrapper(start_points, num_points=num_eps)

    sampled_episodes = []
    for idx in sampled_idx:
        sampled_episodes.append(episodes[idx])

    return sampled_episodes

def sample_episodes_random(episodes, cluster_labels, num_eps):

    sampled_episodes = []
    clusters = np.unique(cluster_labels)
    for cl in clusters:

        cl_eps = [x for (i, x) in enumerate(episodes) if cluster_labels[i] == cl]

        if num_eps < len(cl_eps):
            sampled_episodes.extend(random.sample(cl_eps, k=num_eps))
        else:
            sampled_episodes.extend(cl_eps)

    return sampled_episodes

def visualize_cluster_points(scene, scene_data, episodes, category, cluster_labels, out_path):

    objnav_config = get_objnav_config(scene)
    sim = get_simulator(objnav_config)

    topdown_map = None
    # # visualize goals
    # # import pdb;pdb.set_trace()
    for goal in scene_data['goals_by_category'][f'{scene}_{category}']['goals']:
        topdown_map = _visualize_goal(goal['position'], goal['object_id'], topdown_map, sim, np.array(COLOR_MAP(1.))[:3]*255)

    for i, ep in tqdm(enumerate(episodes)):

        topdown_map = get_topdown_map(
            sim,
            start_pos=ep["start_position"],
            marker="circle",
            color=np.array([255., 0., 0.]),#np.array(COLOR_MAP(cluster_labels[i] / np.max(cluster_labels)))[:3]*255,
            radius=3,
            topdown_map=topdown_map,
            boundary=False,
        )
    
    os.makedirs(out_path, exist_ok=True)
    
    episode_viz_output_filename = os.path.join(out_path, f"{scene}_{category}.jpg")
    cv2.putText(topdown_map, f"Num Episodes: {len(episodes)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imwrite(episode_viz_output_filename, topdown_map[:, :, ::-1])
    import pdb;pdb.set_trace()

def main():

    seed()
    os.makedirs(OUT_PATH)

    scenes = os.listdir(SCENES_PATH)

    for sc in tqdm(scenes, desc='processing scene episodes'):
        with gzip.open(os.path.join(SCENES_PATH, sc), "r") as fin:
            ep_data = json.loads(fin.read().decode("utf-8"))

        # separate episodes category-wise
        cat_sampled_episodes = []
        for cat in ep_data['category_to_task_category_id'].keys():

            cat_eps = sample_episodes_per_cat(ep_data['episodes'], cat)

            if len(cat_eps) == 0:
                continue

            # sample X number of episodes from each category 
            NUM_EPS = int(NUM_PER_SCENE * (len(cat_eps) / len(ep_data['episodes'])))
            cat_eps = sample_episodes_with_fps(cat_eps, num_eps=NUM_EPS)

            if VISUALIZE:
                visualize_cluster_points(sc.replace('.json.gz', ''), ep_data, cat_eps, cat, None, f'{OUT_PATH}/vis')
            cat_sampled_episodes.extend(cat_eps)
        # combine again 
        ep_data['episodes'] = cat_sampled_episodes

        with gzip.open(os.path.join(OUT_PATH, sc), "wt") as fp:
            fp.write(DatasetJSONEncoder().encode(ep_data))

if __name__ == '__main__':
    main()