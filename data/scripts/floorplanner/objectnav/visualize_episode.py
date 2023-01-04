"""
Script to visualize the generated episodes.

To visualize all the goal objects with associated start points run 
python data/scripts/floorplanner/objectnav/visualize_episode.py --in_path /path/to/generated/episodes/folder/ \
            --scene 102344094 --category cabinet --out_path /path/to/output/visualizatios/ 

To visualize all the start points only  
python data/scripts/floorplanner/objectnav/visualize_episode.py --in_path /path/to/generated/episodes/folder/ \
            --scene 102344094 --category cabinet --out_path /path/to/output/visualizatios/ --only_start

To visualize individual episodes with corresponding start and goal points 
python data/scripts/floorplanner/objectnav/visualize_episode.py --in_path /path/to/generated/episodes/folder/ \
            --scene 102344094 --category cabinet --out_path /path/to/output/visualizatios/ --per_episode
"""
import argparse
import gzip
import json
import os
import pickle

import cv2
import tqdm
import GPUtil
import numpy as np
import seaborn as sns

import habitat
import habitat_sim
from data.scripts.floorplanner.utils.utils import (
    COLOR_PALETTE,
    get_topdown_map,
    draw_obj_bbox_on_topdown_map
)
from habitat.config.default import get_config
from habitat.tasks.rearrange.utils import get_aabb
from habitat.datasets.pointnav.pointnav_generator import ISLAND_RADIUS_LIMIT


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

SCENES_ROOT = "data/scene_datasets/floorplanner/v1"

COLOR_MAP = np.array(sns.color_palette("tab10", 30)) * 255 #assuming max 30 goals on a scene TODO: confirm

def get_objnav_config(scene):

    CFG = "configs/tasks/objectnav_fp.yaml"
    SCENE_DATASET_CFG = os.path.join(
        SCENES_ROOT, "hab-fp.scene_dataset_config.json"
    )

    objnav_config = get_config(CFG).clone()
    objnav_config.defrost()
    objnav_config.TASK.SENSORS = []
    objnav_config.SIMULATOR.AGENT_0.SENSORS = ["SEMANTIC_SENSOR", "RGB_SENSOR"]
    FOV = 90
    objnav_config.SIMULATOR.RGB_SENSOR.HFOV = FOV
    objnav_config.SIMULATOR.DEPTH_SENSOR.HFOV = FOV
    objnav_config.SIMULATOR.SEMANTIC_SENSOR.HFOV = FOV

    objnav_config.SIMULATOR.SEMANTIC_SENSOR.WIDTH //= 2
    objnav_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT //= 2
    objnav_config.SIMULATOR.RGB_SENSOR.WIDTH //= 2
    objnav_config.SIMULATOR.RGB_SENSOR.HEIGHT //= 2
    objnav_config.TASK.MEASUREMENTS = []

    deviceIds = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
    )

    deviceId = deviceIds[0]
    objnav_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = deviceId

    objnav_config.SIMULATOR.SCENE = scene
    objnav_config.SIMULATOR.SCENE_DATASET = SCENE_DATASET_CFG
    objnav_config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
    objnav_config.freeze()
    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.SIMULATOR)

    # TODO: precompute navmeshes?
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = objnav_config.SIMULATOR.AGENT_0.RADIUS
    navmesh_settings.agent_height = objnav_config.SIMULATOR.AGENT_0.HEIGHT
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=True
    )
    return sim

def visualize_all_start_points(episodes, scene, category, sim, out_path):

    topdown_map = None

    for ep in episodes["episodes"]:

        cat = ep["object_category"]
        if cat != category:
            continue

        topdown_map = get_topdown_map(
            sim,
            start_pos=ep["start_position"],
            marker="circle",
            color=COLOR_PALETTE["orange"],
            radius=3,
            topdown_map=topdown_map,
        )

    os.makedirs(out_path, exist_ok=True)
    
    episode_viz_output_filename = os.path.join(out_path, f"{scene}_{category}.jpg")
    cv2.imwrite(episode_viz_output_filename, topdown_map[:, :, ::-1])

def _visualize_goal(object_position, object_id, topdown_map, sim, color):

    object_position_on_floor = np.array(object_position).copy()
    while True:
        # TODO: think of better way than ->
        navigable_point = sim.sample_navigable_point()
        if sim.island_radius(navigable_point) >= ISLAND_RADIUS_LIMIT:
            break

    object_position_on_floor[1] = navigable_point[1]

    topdown_map = get_topdown_map(
        sim,
        start_pos=object_position_on_floor,
        topdown_map=topdown_map,
        marker="circle",
        radius=6,
        color=color,
    )
    aabb = get_aabb(object_id, sim, transformed=True)
    topdown_map = draw_obj_bbox_on_topdown_map(topdown_map, aabb, sim)

    return topdown_map

def _visualize_viewpoints(viewpoints, topdown_map, sim, color):

    for p in viewpoints:

        topdown_map = get_topdown_map(
            sim,
            start_pos=p['agent_state']['position'],
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=color,
        )

    return topdown_map

def visualize_all_start_end_points(episodes, scene, category, sim, out_path):

    cat_episodes = [ep for ep in episodes["episodes"] if ep["object_category"] == category]

    topdown_map = None
    goal_set = set([g['object_id'] for g in episodes['goals_by_category'][f'{scene}_{category}']['goals']])
    print(goal_set)
    goal_to_idx = {k: idx for idx, k in enumerate(goal_set)}

    # visualize goals
    for goal in episodes['goals_by_category'][f'{scene}_{category}']['goals']:
        topdown_map = _visualize_goal(goal['position'], goal['object_id'], topdown_map, sim, COLOR_MAP[goal_to_idx[goal['object_id']]][::-1])
        # topdown_map = _visualize_viewpoints(goal['view_points'], topdown_map, sim, COLOR_MAP[goal_to_idx[goal['object_id']]][::-1])

    for ep in tqdm.tqdm(cat_episodes):

        goal_obj_id = ep["info"]["closest_goal_object_id"]

        topdown_map = get_topdown_map(
            sim,
            start_pos=ep["start_position"],
            marker="circle",
            color=COLOR_MAP[goal_to_idx[goal_obj_id]][::-1],
            radius=3,
            topdown_map=topdown_map,
        )
    
    os.makedirs(out_path, exist_ok=True)
    
    episode_viz_output_filename = os.path.join(out_path, f"{scene}_{category}.jpg")
    cv2.imwrite(episode_viz_output_filename, topdown_map[:, :, ::-1])


def visualize_per_episode(episodes, scene, category, sim, out_path):

    cat_episodes = [ep for ep in episodes["episodes"] if ep["object_category"] == category]
    for ep in tqdm.tqdm(cat_episodes):

        goal_obj_id = ep["info"]["closest_goal_object_id"]

        goal = [goal for goal in episodes['goals_by_category'][f'{scene}_{category}']['goals'] if goal['object_id'] == goal_obj_id][0]
        topdown_map = _visualize_goal(goal['position'], goal['object_id'], None, sim, COLOR_PALETTE["red"])
        topdown_map = _visualize_viewpoints(goal['view_points'], topdown_map, sim, COLOR_PALETTE['black'])

        topdown_map = get_topdown_map(
            sim,
            start_pos=ep["start_position"],
            marker="circle",
            color=COLOR_PALETTE["orange"],
            radius=3,
            topdown_map=topdown_map,
        )

        os.makedirs(out_path, exist_ok=True)
        
        episode_viz_output_filename = os.path.join(out_path, f"{scene}_{category}_{ep['episode_id']}.jpg")
        cv2.imwrite(episode_viz_output_filename, topdown_map[:, :, ::-1])    

def main(args):

    scene_path = os.path.join(args.in_path, f'{args.scene}.json.gz')

    with gzip.open(scene_path, 'rt') as fp:
        episodes = json.load(fp)

    scene = args.scene
    category = args.category

    objnav_config = get_objnav_config(scene)
    sim = get_simulator(objnav_config)

    if args.per_episode:
        visualize_per_episode(episodes, scene, category, sim, args.out_path)
    elif args.only_start:
        visualize_all_start_points(episodes, scene, category, sim, args.out_path)
    else:
        visualize_all_start_end_points(episodes, scene, category, sim, args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', required=True, type=str, help="path where episodes and scene goals are placed")
    parser.add_argument('--scene', type=str, required=True, help="scene id")
    parser.add_argument('--category', type=str, default="cabinet")
    parser.add_argument('--only_start', action='store_true', help="generate top down map for with all start points only")
    parser.add_argument('--per_episode', action='store_true', help="generate top down map for each episode with start point and goal")
    parser.add_argument('--out_path', type=str, default='./', help="path to output the topdown maps")
    args = parser.parse_args()
    main(args)