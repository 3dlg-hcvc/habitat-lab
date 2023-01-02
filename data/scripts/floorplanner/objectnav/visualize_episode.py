import argparse
import gzip
import json
import os
import pickle

import cv2
import tqdm
import GPUtil

import habitat
import habitat_sim
from data.scripts.floorplanner.utils.utils import (
    COLOR_PALETTE,
    get_topdown_map,
)
from habitat.config.default import get_config


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

SCENES_ROOT = "data/scene_datasets/floorplanner/v1"

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

def visualize_per_episode(episodes, goals_by_category, scene, category, sim, out_path):

    cat_episodes = [ep for ep in episodes["episodes"] if ep["object_category"] == category]
    for ep in tqdm.tqdm(cat_episodes):

        cat = ep["object_category"]
        if cat != category:
            continue

        goal_obj_id = ep["info"]["closest_goal_object_id"]
        topdown_map = goals_by_category[f'{scene}_{category}']['topdown_maps'][goal_obj_id].copy()

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

    scene_path = os.path.join(args.in_path, f'content/{args.scene}.json.gz')
    goals_path = os.path.join(args.in_path, f'scene_goals/{args.scene}_goal_objs.pkl')

    with gzip.open(scene_path, 'rt') as fp:
        episodes = json.load(fp)

    goals_by_category = None
    if os.path.exists(goals_path):
        with open(goals_path, "rb") as f:
            goals_by_category = pickle.load(f)

    scene = args.scene
    category = args.category

    objnav_config = get_objnav_config(scene)
    sim = get_simulator(objnav_config)

    if args.per_episode:
        if goals_by_category is None:
            raise Exception(f"Scene goals not found at {goals_path}")
        visualize_per_episode(episodes, goals_by_category, scene, category, sim, args.out_path)
    else:
        visualize_all_start_points(episodes, scene, category, sim, args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', required=True, type=str, help="path where episodes and scene goals are placed")
    parser.add_argument('--scene', type=str, required=True, help="scene id")
    parser.add_argument('--category', type=str, default='cabinet')
    parser.add_argument('--per_episode', action='store_true', help="generate top down map for each episode with start point and goal")
    parser.add_argument('--out_path', type=str, default='./', help="path to output the topdown maps")
    args = parser.parse_args()
    main(args)