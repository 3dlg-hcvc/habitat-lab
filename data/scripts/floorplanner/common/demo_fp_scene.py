import cv2

import habitat
from data.scripts.floorplanner.utils.utils import get_topdown_map
from habitat.config import read_write
from habitat_sim.nav import NavMeshSettings

task_config_path = (
    "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_fp.yaml"
)


def visualize_fp_scenes(scenes):
    for scene in scenes:
        cfg = habitat.get_config(task_config_path)
        with read_write(cfg):
            cfg.habitat.simulator.scene = scene

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

        pos = sim.pathfinder.get_random_navigable_point()
        # pos = [-6.7892866 ,  0.19999951, -4.6983275 ]
        rot = sim.get_agent_state().rotation
        sim.set_agent_state(pos, rot)

        topdown_map = get_topdown_map(sim, pos, rot)
        cv2.imwrite(f"ego_obs_{scene}.png", sim.render()[:, :, ::-1])
        cv2.imwrite(f"topdown_map_{scene}.png", topdown_map)

        sim.close(destroy=True)


if __name__ == "__main__":
    scenes = ["102344349"]
    visualize_fp_scenes(scenes)
