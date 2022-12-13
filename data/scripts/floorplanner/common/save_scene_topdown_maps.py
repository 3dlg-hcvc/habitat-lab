import os

import cv2
from tqdm import tqdm

import habitat
from data.scripts.floorplanner.utils.utils import get_topdown_map
from habitat.utils.visualizations import maps
from habitat_sim.nav import NavMeshSettings

scenes_path = (
    "data/scene_datasets/floorplanner/v1/assets/compressed-glb-arch-only"
)
dataset_config_path = (
    "data/scene_datasets/floorplanner/v1/hab-fp.scene_dataset_config.json"
)
cfg_path = "configs/robots/stretch.yaml"
topdown_map_out_path = "data/scene_datasets/floorplanner/v1/viz/topdown_maps"
os.makedirs(topdown_map_out_path, exist_ok=True)

NUM_TRIES = 20


def save_topdown_maps(scenes):
    for scene in tqdm(scenes):
        topdown_map_path = os.path.join(topdown_map_out_path, f"{scene}.jpg")

        if os.path.exists(topdown_map_path):
            continue

        cfg = habitat.get_config(cfg_path)
        cfg.defrost()
        cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
        cfg.SIMULATOR.SCENE = scene
        cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = cfg.SIMULATOR.AGENT_0.RADIUS
        navmesh_settings.agent_height = cfg.SIMULATOR.AGENT_0.HEIGHT

        sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=True
        )

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
