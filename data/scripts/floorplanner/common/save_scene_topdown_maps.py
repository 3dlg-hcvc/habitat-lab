import os

import cv2
from tqdm import tqdm

import habitat
from data.scripts.floorplanner.utils.utils import get_topdown_map
from habitat.config import read_write
from habitat.utils.visualizations import maps

scenes_path = (
    "data/scene_datasets/floorplanner/v1/assets/compressed-glb-arch-only"
)
task_config_path = (
    "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_fp.yaml"
)
topdown_map_out_path = "data/scene_datasets/floorplanner/v1/viz/topdown_maps"
os.makedirs(topdown_map_out_path, exist_ok=True)

NUM_TRIES = 10


def save_topdown_maps(scenes):
    for scene in tqdm(scenes):
        topdown_map_path = os.path.join(topdown_map_out_path, f"{scene}.jpg")

        if os.path.exists(topdown_map_path):
            continue

        cfg = habitat.get_config(task_config_path)

        with read_write(cfg):
            cfg.habitat.simulator.scene = scene

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

        max_nav_area_pos = None
        max_nav_area = 0

        # getting the topdown map with the most nav area visible
        for i in tqdm(range(NUM_TRIES)):
            pos = sim.pathfinder.get_random_navigable_point()

            nav_area = maps.get_topdown_map(
                sim.pathfinder, height=pos[1]
            ).sum()
            if nav_area > max_nav_area:
                max_nav_area = nav_area
                max_nav_area_pos = pos

        rot = sim.get_agent_state().rotation
        sim.set_agent_state(max_nav_area_pos, rot)

        topdown_map = get_topdown_map(sim, marker=None)
        cv2.imwrite(topdown_map_path, topdown_map[:, :, ::-1])
        sim.close(destroy=True)


if __name__ == "__main__":
    scenes = [x.split(".")[0] for x in os.listdir(scenes_path)]
    save_topdown_maps(scenes)
