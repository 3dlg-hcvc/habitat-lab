import math

import magnum as mn
import numpy as np

import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils


def get_topdown_map(sim, start_pos=None, start_rot=None):
    if start_pos is None:
        start_pos = sim.get_agent_state().position
    if start_rot is None:
        start_rot = sim.get_agent_state().rotation

    topdown_map = maps.get_topdown_map(sim.pathfinder, height=start_pos[1])

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    topdown_map = recolor_map[topdown_map]
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])

    # convert world agent position to maps module grid point
    agent_grid_pos_source = maps.to_grid(
        start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )

    agent_grid_pos_source = maps.to_grid(
        start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )
    agent_forward = utils.quat_to_magnum(
        sim.get_agent_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))
    agent_forward = utils.quat_to_magnum(
        sim.agents[0].get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))

    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    maps.draw_agent(
        topdown_map,
        agent_grid_pos_source,
        agent_orientation,
        agent_radius_px=24,
    )

    return topdown_map


def get_topdown_map_with_path(sim, start_pos, start_rot, goal_pos):
    topdown_map = get_topdown_map(sim, start_pos, start_rot)
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])

    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos

    assert sim.pathfinder.find_path(path)

    # print(f"geodesic_distance: {path.geodesic_distance})

    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path.points
    ]
    # draw the trajectory on the map
    maps.draw_path(topdown_map, trajectory)

    return topdown_map
