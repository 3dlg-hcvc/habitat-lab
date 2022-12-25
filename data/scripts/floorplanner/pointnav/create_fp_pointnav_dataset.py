"""
Script for creating a PointNav dataset for Floor-planner scenes (sourced from habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py).
"""

import argparse
import gzip
import json
import multiprocessing
import os
from itertools import repeat

import cv2
import yaml
from tqdm import tqdm
import pdb

import habitat
from data.scripts.floorplanner.utils.utils import get_topdown_map_with_path
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)
from habitat_sim.nav import NavMeshSettings

NUM_EPISODES_PER_SCENE = int(1)

splits_info_path = "data/scene_datasets/ai2thor-hab/scene_splits.yaml"
dataset_config_path = (
    "data/scene_datasets/ai2thor-hab/ai2thor.scene_dataset_config.json"
)

output_dataset_path = "data/datasets/pointnav/ai2thor-hab/v0"


def _generate_fn(scene, split, args):
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
    pdb.set_trace()
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE, is_gen_shortest_path=False
        )
    )
    pdb.set_trace()
    if args.viz:
        ep = dset.episodes[0]
        viz_out_file = os.path.join(
            output_dataset_path,
            split,
            "viz",
            f"topdown_scene={scene}_ep={ep.episode_id}.jpg",
        )
        topdown_map = get_topdown_map_with_path(
            sim, ep.start_position, ep.start_rotation, ep.goals[0].position
        )
        cv2.imwrite(viz_out_file, topdown_map)

    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    sim.close()


def generate_fp_pointnav_dataset(args):
    # Load splits
    with open(splits_info_path, "r") as f:
        scene_splits = yaml.safe_load(f)

    if args.split == "all":
        splits = scene_splits.keys()
    else:
        splits = [args.split]

    for split in splits:
        print(f"Creating {split} dataset.")

        os.makedirs(
            os.path.join(output_dataset_path, split, "content"), exist_ok=True
        )

        if args.viz:
            os.makedirs(
                os.path.join(output_dataset_path, split, "viz"), exist_ok=True
            )

        scenes = scene_splits[split]

        # with multiprocessing.Pool(4) as pool, tqdm(total=len(scenes)) as pbar:
        #     for _ in pool.starmap(
        #         _generate_fn, zip(scenes, repeat(split), repeat(args))
        #     ):
        #         pbar.update()

        # [for debugging]
        for scene in tqdm(scenes):
            _generate_fn(scene, split, args)
            pdb.set_trace()

        path = os.path.join(output_dataset_path, split, split + ".json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with gzip.open(path, "wt") as f:
            json.dump(dict(episodes=[]), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "val", "all"],
        default="all",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Visualize topdown map for first episode of each scene.",
    )
    args = parser.parse_args()

    generate_fp_pointnav_dataset(args)
