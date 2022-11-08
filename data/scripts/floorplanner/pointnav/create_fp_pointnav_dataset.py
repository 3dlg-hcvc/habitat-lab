"""
Script for creating a PointNav dataset for Floor-planner scenes (sourced from habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py).
"""

import gzip
import json
import multiprocessing
import os
from itertools import repeat

import tqdm
import yaml

import habitat
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)
from habitat_sim.nav import NavMeshSettings

NUM_EPISODES_PER_SCENE = int(100)

splits_info_path = "data/scene_datasets/floorplanner/v1/scene_splits.yaml"
dataset_config_path = (
    "data/scene_datasets/floorplanner/v1/hab-fp.scene_dataset_config.json"
)

output_dataset_path = "data/datasets/pointnav/floorplanner/v1"


def _generate_fn(scene, split):
    out_file = os.path.join(
        output_dataset_path, split, "content", f"{scene}.json.gz"
    )

    if os.path.exists(out_file):
        return

    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=True
    )

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE, is_gen_shortest_path=False
        )
    )

    for ep in dset.episodes:
        ep.scene_id = scene

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    sim.close()


def generate_fp_pointnav_dataset():
    os.makedirs(output_dataset_path, exist_ok=True)

    # Load splits
    with open(splits_info_path, "r") as f:
        scene_splits = yaml.safe_load(f)

    for split in scene_splits.keys():
        scenes = scene_splits[split]

        with multiprocessing.Pool(4) as pool, tqdm.tqdm(
            total=len(scenes)
        ) as pbar:
            for _ in pool.starmap(_generate_fn, zip(scenes, repeat(split))):
                pbar.update()

        # [DEBUG]
        # for scene in scenes:
        #     _generate_fn(scene, split)

        path = os.path.join(output_dataset_path, split, split + ".json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with gzip.open(path, "wt") as f:
            json.dump(dict(episodes=[]), f)


if __name__ == "__main__":
    generate_fp_pointnav_dataset()
