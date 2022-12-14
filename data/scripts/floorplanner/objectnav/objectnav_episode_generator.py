#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from typing import List

import cv2
import imageio
import numpy as np

import habitat_sim
from data.scripts.floorplanner.utils.utils import (
    COLOR_PALETTE,
    draw_obj_bbox_on_topdown_map,
    get_topdown_map,
)
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.pointnav.pointnav_generator import (
    ISLAND_RADIUS_LIMIT,
    _ratio_sample_rate,
)
from habitat.datasets.utils import get_action_shortest_path
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_to_angle_axis,
    quat_to_coeffs,
)


def _direction_to_quaternion(direction_vector: np.array):
    origin_vector = np.array([0, 0, -1])
    output = quaternion_from_two_vectors(origin_vector, direction_vector)
    output = output.normalized()
    return output


def _get_multipath(sim: HabitatSim, start, ends):
    multi_goal = habitat_sim.MultiGoalShortestPath()
    multi_goal.requested_start = start
    multi_goal.requested_ends = ends
    sim.pathfinder.find_path(multi_goal)
    return multi_goal


def _get_action_shortest_path(
    sim: HabitatSim, start_pos, start_rot, goal_pos, goal_radius=0.05
):
    sim.set_agent_state(start_pos, start_rot, reset_sensors=True)
    greedy_follower = sim.make_greedy_follower()
    return greedy_follower.find_path(goal_pos)


# TODO: cleaning
def is_compatible_episode(
    source_position,
    target_positions,
    sim: HabitatSim,
    goals: List[ObjectGoal],
    near_dist,
    far_dist,
    geodesic_to_euclid_ratio,
    same_floor_flag=False,
):
    FAIL_TUPLE = False, 0, 0, [], [], [], []
    if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
        print("=====> Island radius failure")
        return FAIL_TUPLE
    s = np.array(source_position)

    # TWO TARGETS MAY BE BETWEEN TWO GOALS
    goal_targets = [
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    ]

    if same_floor_flag:
        valid = False
        for gt in goal_targets:
            gt = np.array(gt)
            valid = np.any(np.abs(gt[:, 1] - s[1]) < 0.5)
            if valid:
                break
        if not valid:
            return FAIL_TUPLE

    closest_goal_targets = (
        sim.geodesic_distance(s, vps) for vps in goal_targets
    )
    closest_goal_targets, goals_sorted = zip(
        *sorted(zip(closest_goal_targets, goals), key=lambda x: x[0])
    )
    d_separation = closest_goal_targets[0]

    if (
        np.isinf(d_separation)
        # np.inf in closest_goal_targets
        or not near_dist <= d_separation <= far_dist
    ):
        # print('=====> Distance threshold failure: {}, {}, {}, {}'.format(
        #     near_dist, d_separation, far_dist, np.inf in closest_goal_targets
        # ))
        return FAIL_TUPLE

    # shortest_path = sim.get_straight_shortest_path_points(s, closest_target)
    shortest_path = None
    # shortest_path = closest_goal_targets[0].points
    euclid_dist = np.linalg.norm(s - goals_sorted[0].position)
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        # print(f'=====> Distance ratios: {distances_ratio} {geodesic_to_euclid_ratio}')
        return FAIL_TUPLE

    # geodesic_distances, _ = zip(*closest_goal_targets)
    geodesic_distances = closest_goal_targets

    # euclid_distances_target = np.linalg.norm(np.array([goal.position for goal in goals]) - shortest_path[-1:], axis=1)
    # _, goals_sorted = zip(*sorted(zip(euclid_distances_target, goals)))
    # goal_index = np.argmin(euclid_distances_target)
    # goals_sorted = [goals[goal_index]] + goals[:goal_index] + goals[goal_index+1:]

    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [
        0,
        np.sin(angle / 2),
        0,
        np.cos(angle / 2),
    ]  # Pick random starting rotation

    # try:
    #     action_shortest_path = _get_action_shortest_path(
    #         sim, s, source_rotation, shortest_path[-1]
    #     )
    #     if action_shortest_path == None:
    #         return FAIL_TUPLE
    #     shortest_path = (
    #         action_shortest_path
    #     )
    #     # [ShortestPathPoint(point, [0,0,0], action) for point, action in zip(shortest_path, action_shortest_path)]
    # except GreedyFollowerError:
    #     print(
    #         "Could not find path between %s and %s"
    #         % (str(s), str(shortest_path[-1]))
    #     )
    #     return FAIL_TUPLE
    ending_state = None
    # ending_state = closest_goal_targets[0].points[-1]  # sim.get_agent_state()
    # ending_position = ending_state.position
    #
    # e_q = ending_state.rotation  # We presume the agent is upright
    # # print(shortest_path)
    # goal_direction = goals_sorted[0].position - ending_position
    # goal_direction[1] = 0
    # a_q = _direction_to_quaternion(goal_direction)
    # quat_delta = e_q - a_q
    # # The angle between the two quaternions should be how much to turn
    # theta, _ = quat_to_angle_axis(quat_delta)
    # if theta < 0 or theta > np.pi:
    #     turn = HabitatSimActions.TURN_LEFT
    #     if theta > np.pi:
    #         theta = 2 * np.pi - theta
    # else:
    #     turn = HabitatSimActions.TURN_RIGHT
    # turn_angle = np.deg2rad(sim.config.TURN_ANGLE)
    # num_of_turns = int(theta / turn_angle + 0.5)
    # shortest_path += [turn] * num_of_turns
    # angle = angle_between_quaternions(a_q, e_q)

    # if len(shortest_path) > 750:
    #     print("ERROR SHORTEST PATH IS TOO LONG")
    #     return FAIL_TUPLE

    # shortest_path = get_action_shortest_path(sim,
    #     s,
    #     source_rotation,
    #     goals_sorted[0].position,
    #     max_episode_steps=750)
    # #CANNOT CATCH ERRORS
    # if shortest_path == None:
    #     return FAIL_TUPLE
    # Make it so it doesn't have to see the object initially
    # TODO Consider turning this check back on later?
    # num_rotation_attempts = 20
    # for attempt in range(num_rotation_attempts):
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

    #     obs = sim.get_observations_at(s, source_rotation)

    #     if all(
    #         (compute_pixel_coverage(obs["semantic"], g.object_id) / g.best_iou) < 0.01
    #         for g in goals_sorted
    #     ):
    #         break

    # if attempt == num_rotation_attempts:
    #     return FAIL_TUPLE

    return (
        True,
        d_separation,
        euclid_dist,
        source_rotation,
        geodesic_distances,
        goals_sorted,
        shortest_path,
        ending_state,
    )


def build_goal(
    sim: HabitatSim,
    object_id: int,
    object_name_id: str,
    object_category_name: str,
    object_category_id: int,
    object_position,
    object_aabb: habitat_sim.geo.BBox,
    object_obb: habitat_sim.geo.OBB,
    cell_size: float = 1.0,
    grid_radius: float = 10.0,
    turn_radians: float = np.pi / 9,
    max_distance: float = 1.0,
):
    object_position = object_aabb.center()
    eps = 1e-5
    x_len, _, z_len = np.array(object_aabb.size()) / 2.0 + max_distance

    x_bxp = np.arange(-x_len, x_len + eps, step=cell_size) + object_position[0]
    z_bxp = np.arange(-z_len, z_len + eps, step=cell_size) + object_position[2]

    # poses from where the goal object might be visible
    candidate_poses = [
        np.array([x, object_position[1], z])
        for x, z in itertools.product(x_bxp, z_bxp)
    ]

    def down_is_navigable(pt, search_dist=2.0):
        pf = sim.pathfinder
        delta_y = 0.05
        max_steps = int(search_dist / delta_y)
        step = 0
        pt_tmp = pt.copy()
        is_navigable = pf.is_navigable(pt, 2)

        while not is_navigable:
            pt_tmp[1] -= delta_y
            is_navigable = pf.is_navigable(pt_tmp)
            step += 1
            if step == max_steps:
                return False
        return True

    def _get_iou(x, y, z):
        pt = np.array([x, y, z])
        if not object_obb.distance(pt) <= max_distance:
            return (
                -1,
                pt,
                None,
                "too_far",
            )  # viewpoint too far from goal object
        if not down_is_navigable(pt):
            return (
                -1,
                pt,
                None,
                "down_unnavigable",
            )  # point below the viewpoint is not navigable

        pf = sim.pathfinder
        pt = np.array(pf.snap_point(pt))

        goal_direction = object_position - pt

        goal_direction[1] = 0

        q = _direction_to_quaternion(goal_direction)

        cov = 0
        agent = sim.get_agent(0)

        # UPDATE: simpler sampling of observations from look_up and look_down
        sim.set_agent_state(pt, q)
        for act in [
            HabitatSimActions.LOOK_DOWN,
            HabitatSimActions.LOOK_UP,
            HabitatSimActions.LOOK_UP,
        ]:
            agent.act(act)
            # UPDATE: simpler sampling of observations from look_up and look_down
            obs = sim.render("semantic")

            # for v in agent._sensors.values():
            #     v.set_transformation_from_spec()
            #     obs = sim.get_observations_at(
            #         pt, q, keep_agent_at_new_pose=True
            #     )
            # cov += compute_pixel_coverage(obs["semantic"], object_id)

            cov += compute_pixel_coverage(obs, object_id)
        return cov, pt, q, "Success"

    def _visualize_rejected_viewpoints(x, y, z, iou):
        pt = np.array([x, y, z])

        pt[1] -= 0.5
        pf = sim.pathfinder
        pt = np.array(pf.snap_point(pt))

        goal_direction = object_position - pt
        goal_direction[1] = 0

        q = _direction_to_quaternion(goal_direction)

        obs = sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
        rejected_out_path = "data/images/rejected"
        os.makedirs(rejected_out_path, exist_ok=True)
        imageio.imsave(
            os.path.join(
                rejected_out_path,
                f"rejected_{object_category_name}_{object_name_id}_{object_id}_{iou}_.png",
            ),
            observations_to_image(obs, info={}),
        )
        sem = obs["semantic"]
        sem[sem != object_id] = 0
        sem = sem * 255 / object_id
        sem = np.stack((sem[..., 0],) * 3, axis=-1)
        rgb_sem_concat = np.hstack((sem, observations_to_image(obs, info={})))
        imageio.imsave(
            os.path.join(
                rejected_out_path,
                f"{object_category_name}_{object_name_id}_{object_id}_{iou}_.png",
            ),
            rgb_sem_concat,
        )

    candidate_poses_ious_orig = list(_get_iou(*pos) for pos in candidate_poses)
    n_orig_poses = len(candidate_poses_ious_orig)
    n_too_far_rejected = 0
    n_down_not_navigable_rejected = 0
    for p in candidate_poses_ious_orig:
        if p[-1] == "too_far":
            n_too_far_rejected += 1
        elif p[-1] == "down_unnavigable":
            n_down_not_navigable_rejected += 1
    candidate_poses_ious_orig_2 = [
        p for p in candidate_poses_ious_orig if p[0] > 0
    ]

    # Reject candidate_poses that do not satisfy island radius constraints
    candidate_poses_ious = [
        p
        for p in candidate_poses_ious_orig_2
        if sim.island_radius(p[1]) >= ISLAND_RADIUS_LIMIT
    ]
    n_island_candidates_rejected = len(candidate_poses_ious_orig_2) - len(
        candidate_poses_ious
    )
    best_iou = (
        max(v[0] for v in candidate_poses_ious)
        if len(candidate_poses_ious) != 0
        else 0
    )

    error_info_str = []
    # [DEBUG]
    if best_iou <= 0.00:
        ntot = n_orig_poses
        nic = n_island_candidates_rejected
        nur = n_too_far_rejected
        ndn = n_down_not_navigable_rejected
        error_info_str = [f"-------- Object ID: {object_name_id}_{object_id}:"]
        error_info_str.append(f"[{nic}/{ntot}] rejected due to island radius")
        error_info_str.append(f"[{nur}/{ntot}] rejected due to too far")
        error_info_str.append(
            f"[{ndn}/{ntot}] rejected because the surface below is unnavigable"
        )

        print("\n".join(error_info_str))

    # [UPDATE]
    keep_thresh = 0.001

    # [DEBUG]: visualize views from rejected viewpoints
    #     for p in candidate_poses_ious:
    #         if p[0] <= keep_thresh:
    #             x, y, z = p[1].tolist()
    #             _visualize_rejected_viewpoints(x, y, z, p[0])

    view_locations = [
        ObjectViewLocation(
            AgentState(pt.tolist(), quat_to_coeffs(q).tolist()), iou
        )
        for iou, pt, q, _ in candidate_poses_ious
        if iou is not None and iou > keep_thresh
    ]
    island_view_locations = [
        ObjectViewLocation(AgentState(pt.tolist()), q)
        for iou, pt, q, _ in candidate_poses_ious_orig_2
        if sim.island_radius(pt) < ISLAND_RADIUS_LIMIT
    ]
    iou_rejected_view_locations = [
        ObjectViewLocation(AgentState(pt.tolist()), q)
        for iou, pt, q, _ in candidate_poses_ious_orig
        if iou <= keep_thresh and iou >= 0
    ]
    too_far_view_locations = [
        ObjectViewLocation(AgentState(pt.tolist()), q)
        for iou, pt, q, reason in candidate_poses_ious_orig
        if iou < 0 and reason == "too_far"
    ]
    down_unnavigable_view_locations = [
        ObjectViewLocation(AgentState(pt.tolist()), q)
        for iou, pt, q, reason in candidate_poses_ious_orig
        if iou < 0 and reason == "down_unnavigable"
    ]

    view_locations = sorted(view_locations, reverse=True, key=lambda v: v.iou)

    # [DEBUG]: visualize approved and rejected viewpoints on topdown maps
    object_position_on_floor = np.array(object_position).copy()
    if len(view_locations) > 0:
        object_position_on_floor[1] = view_locations[0].agent_state.position[1]
    else:
        while True:
            navigable_point = sim.sample_navigable_point()
            if sim.island_radius(navigable_point) >= ISLAND_RADIUS_LIMIT:
                break

        object_position_on_floor[1] = navigable_point[1]

    topdown_map = get_topdown_map(
        sim, start_pos=object_position_on_floor, marker=None
    )

    for view in view_locations:  # all valid points in green
        topdown_map = get_topdown_map(
            sim,
            start_pos=view.agent_state.position,
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=COLOR_PALETTE["green"],
        )
    for view in island_view_locations:  # island points in black
        topdown_map = get_topdown_map(
            sim,
            start_pos=view.agent_state.position,
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=COLOR_PALETTE["black"],
        )
    for view in iou_rejected_view_locations:  # iou rejected points in red
        topdown_map = get_topdown_map(
            sim,
            start_pos=view.agent_state.position,
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=COLOR_PALETTE["red"],
        )
    for view in too_far_view_locations:  # too far errors in blue
        topdown_map = get_topdown_map(
            sim,
            start_pos=view.agent_state.position,
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=COLOR_PALETTE["lighter_blue"],
        )
    for view in down_unnavigable_view_locations:  # too far errors in blue
        topdown_map = get_topdown_map(
            sim,
            start_pos=view.agent_state.position,
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=COLOR_PALETTE["yellow"],
        )

    topdown_map = get_topdown_map(
        sim,
        start_pos=object_position_on_floor,
        topdown_map=topdown_map,
        marker="circle",
        radius=6,
        color=COLOR_PALETTE["red"],
    )
    topdown_map = draw_obj_bbox_on_topdown_map(topdown_map, object_aabb, sim)

    if best_iou <= 0 or len(view_locations) == 0:
        print(
            f"No valid views found for {object_category_name} {object_name_id}_{object_id} in {sim.habitat_config.SCENE}: {best_iou}"
        )
        h = topdown_map.shape[0]
        topdown_map = cv2.copyMakeBorder(
            topdown_map,
            0,
            125,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_PALETTE["white"],
        )
        error_info_str.append(
            f"[{len(iou_rejected_view_locations)}/{n_orig_poses}] rejected due to IoU visibility."
        )
        for i, line in enumerate(error_info_str):
            if "too far" in line:
                color = COLOR_PALETTE["lighter_blue"]
            elif "unnavigable" in line:
                color = COLOR_PALETTE["yellow"]
            elif "IoU visibility" in line:
                color = COLOR_PALETTE["red"]
            else:
                color = COLOR_PALETTE["black"]

            topdown_map = cv2.putText(
                topdown_map,
                line,
                (10, h + 25 * i + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return None, topdown_map

    # [DEBUG]: visualize views from approved viewpoints
    # for view in view_locations:
    #     obs = sim.get_observations_at(
    #         view.agent_state.position, view.agent_state.rotation
    #     )

    #     rgb_obs = np.ascontiguousarray(obs['rgb'][..., :3])
    #     sem_obs = (obs["semantic"] == object_id).astype(np.uint8) * 255
    #     contours, _ = cv2.findContours(sem_obs, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #     obs["rgb"] = cv2.drawContours(rgb_obs, contours, -1, (0, 255, 0), 4)

    #     # obs["rgb"] = get_image_with_obj_overlay(obs, [object_id])
    #     imageio.imsave(
    #        os.path.join(
    #            "data/images/objnav_dataset_gen/",
    #            f"{object_name_id}_{object_id}_{view.iou}_{view.agent_state.position}.png",
    #        ),
    #        observations_to_image(obs, info={}).astype(np.uint8),
    #     )

    goal = ObjectGoal(
        position=np.array(object_position).tolist(),
        view_points=view_locations,
        object_id=object_id,
        object_category=object_category_name,
        object_name=object_name_id,
    )

    return goal, topdown_map


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    goals,
    shortest_paths=None,
    scene_state=None,
    info=None,
    scene_dataset_config="default",
):
    return ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        object_category=goals[0].object_category,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
        scene_dataset_config=scene_dataset_config,
    )


def generate_objectnav_episode(
    sim: HabitatSim,
    goals: List[ObjectGoal],
    scene_state=None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_target: int = 1000,
    same_floor_flag: bool = False,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    :param same_floor_flag should object exist on same floor as agent's start?
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        # target_positions = (
        #     pydash.chain()
        #     .map(lambda g: g.view_points)
        #     .flatten().map(lambda v: v.agent_state.position)(goals)
        # )
        # Cache this transformation
        target_positions = np.array(
            list(
                itertools.chain(
                    *(
                        (
                            view_point.agent_state.position
                            for view_point in g.view_points
                        )
                        for g in goals
                    )
                )
            )
        )

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            if (
                source_position is None
                or np.any(np.isnan(source_position))
                or not sim.is_navigable(source_position)
            ):
                raise RuntimeError("Unable to find valid starting location")
            if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                # print(f'=====> Failed island radius: {sim.island_radius(source_position)} / {ISLAND_RADIUS_LIMIT}')
                continue
            compat_outputs = is_compatible_episode(
                source_position,
                target_positions,
                sim,
                goals,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                same_floor_flag=same_floor_flag,
            )
            is_compatible = compat_outputs[0]
            # if not is_compatible:
            # print('=====> Failed compatibility test')

            if is_compatible:
                (
                    is_compatible,
                    dist,
                    euclid_dist,
                    source_rotation,
                    geodesic_distances,
                    goals_sorted,
                    shortest_path,
                    ending_state,
                ) = compat_outputs
                if shortest_path is None:
                    shortest_paths = None
                else:
                    shortest_paths = [shortest_path]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=sim.habitat_config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=shortest_paths,
                    scene_state=scene_state,
                    info={
                        "geodesic_distance": dist,
                        "euclidean_distance": euclid_dist,
                        "closest_goal_object_id": goals_sorted[0].object_id,
                        # "navigation_bounds": sim.pathfinder.get_bounds(),
                        # "best_viewpoint_position": ending_state.position,
                    },
                    goals=goals_sorted,
                )

                episode_count += 1
                yield episode
                break

        if retry == number_retries_per_target - 1:
            raise RuntimeError("Unable to find valid starting location")


def generate_objectnav_episode_v2(
    sim: HabitatSim,
    goals: List[ObjectGoal],
    cluster_centers: List[List[float]],
    distance_to_clusters: np.float32,
    scene_state=None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_cluster: int = 1000,
    scene_dataset_config: str = "default",
    same_floor_flag: bool = False,
    eps_generated: int = 0,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    :param same_floor_flag should object exist on same floor as agent's start?
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    assert num_episodes > 0
    # cache this transformation
    target_positions = np.array(
        list(
            itertools.chain(
                *(
                    (
                        view_point.agent_state.position
                        for view_point in g.view_points
                    )
                    for g in goals
                )
            )
        )
    )
    ############################################################################
    # Filter out invalid clusters
    ############################################################################
    valid_mask = (distance_to_clusters >= closest_dist_limit) & (
        distance_to_clusters <= furthest_dist_limit
    )
    if same_floor_flag:
        # # [UPDATE] Ensure that cluster is on same floor as ANY object viewpoint
        for i, cluster_info in enumerate(cluster_centers):
            valid_mask[i] = valid_mask[i] & np.any(
                np.abs(cluster_info["center"][1] - target_positions[:, 1])
                < 0.5
            )

    valid_clusters = []
    for i in range(len(cluster_centers)):
        if valid_mask[i].item():
            valid_clusters.append(cluster_centers[i])

    if len(valid_clusters) == 0:
        # visualizing clusters
        # topdown_map = get_topdown_map(sim, marker=None)
        # for c in cluster_centers:
        #     topdown_map = get_topdown_map(sim, start_pos=c['center'], marker='circle', radius=2, color=COLOR_PALETTE['green'], topdown_map=topdown_map)
        raise RuntimeError(
            f"No valid clusters: {len(valid_clusters)}/{len(cluster_centers)}"
        )
    cluster_centers = valid_clusters

    NC = len(cluster_centers)
    ############################################################################
    # Divide episodes across clusters
    ############################################################################
    episodes_per_cluster = np.zeros((len(cluster_centers),), dtype=np.int32)
    if NC <= num_episodes:
        # Case 1: There are more episodes than clusters
        ## Divide episodes equally across clusters
        episodes_per_cluster[:] = num_episodes // NC
        ## Place the residual episodes into random clusters
        residual_episodes = num_episodes - NC * (num_episodes // NC)
        if residual_episodes > 0:
            random_order = np.random.permutation(NC)
            for i in random_order[:residual_episodes]:
                episodes_per_cluster[i] += 1
    else:
        # Case 2: There are fewer episodes than clusters
        ## Sample one episode per cluster for a random subset of clusters.
        random_order = np.random.permutation(NC)
        for i in random_order[:num_episodes]:
            episodes_per_cluster[i] = 1

    ############################################################################
    # Generate episodes for each cluster
    ############################################################################
    episode_id = eps_generated
    for i, num_cluster_episodes in enumerate(episodes_per_cluster):
        episode_count = 0
        cluster_center = cluster_centers[i]["center"]
        cluster_radius = max(3 * cluster_centers[i]["stddev"], 2.0)
        while (
            episode_count < num_cluster_episodes and num_cluster_episodes > 0
        ):
            for _ in range(number_retries_per_cluster):
                source_position = (
                    sim.pathfinder.get_random_navigable_point_near(
                        cluster_center, cluster_radius, max_tries=10000
                    )
                )
                if (
                    source_position is None
                    or np.any(np.isnan(source_position))
                    or not sim.is_navigable(source_position)
                ):
                    print(f"Skipping cluster {cluster_center}")
                    num_cluster_episodes = 0
                    break
                if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                    continue
                compat_outputs = is_compatible_episode(
                    source_position,
                    target_positions,
                    sim,
                    goals,
                    near_dist=closest_dist_limit,
                    far_dist=furthest_dist_limit,
                    geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                    same_floor_flag=same_floor_flag,
                )
                is_compatible = compat_outputs[0]

                if is_compatible:
                    (
                        is_compatible,
                        dist,
                        euclid_dist,
                        source_rotation,
                        geodesic_distances,
                        goals_sorted,
                        shortest_path,
                        ending_state,
                    ) = compat_outputs
                    if shortest_path is None:
                        shortest_paths = None
                    else:
                        shortest_paths = [shortest_path]
                    episode = _create_episode(
                        episode_id=episode_id,
                        scene_id=sim.habitat_config.SCENE,
                        start_position=source_position,
                        start_rotation=source_rotation,
                        shortest_paths=shortest_paths,
                        scene_state=scene_state,
                        info={
                            "geodesic_distance": dist,
                            "euclidean_distance": euclid_dist,
                            "closest_goal_object_id": goals_sorted[
                                0
                            ].object_id,
                        },
                        goals=goals_sorted,
                        scene_dataset_config=scene_dataset_config,
                    )

                    episode_count += 1
                    episode_id += 1
                    yield episode
                    break


def update_objectnav_episode_v2(
    sim: HabitatSim,
    goals: List[ObjectGoal],
    episode: ObjectGoalNavEpisode,
):
    r"""Updates an existing episode with the goals.
    :param sim: simulator with loaded scene for generation.
    """
    ############################################################################
    # Compute distances
    ############################################################################
    source_position = episode.start_position
    source_position = np.array(source_position)
    goal_targets = [
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    ]
    closest_goal_targets = (
        sim.geodesic_distance(source_position, vps) for vps in goal_targets
    )
    closest_goal_targets, goals_sorted = zip(
        *sorted(zip(closest_goal_targets, goals), key=lambda x: x[0])
    )
    d_separation = closest_goal_targets[0]
    shortest_path = None
    euclid_dist = np.linalg.norm(source_position - goals_sorted[0].position)
    ############################################################################
    # Create new episode with updated information
    ############################################################################
    if shortest_path is None:
        shortest_paths = None
    else:
        shortest_paths = [shortest_path]
    episode_new = _create_episode(
        episode_id=episode.episode_id,
        scene_id=sim.habitat_config.SCENE,
        start_position=episode.start_position,
        start_rotation=episode.start_rotation,
        shortest_paths=shortest_paths,
        scene_state=None,
        info={
            "geodesic_distance": d_separation,
            "euclidean_distance": euclid_dist,
            "closest_goal_object_id": goals_sorted[0].object_id,
        },
        goals=goals_sorted,
        scene_dataset_config=episode.scene_dataset_config,
    )
    return episode_new


def generate_objectnav_episode_with_added_objects(
    sim: HabitatSim,
    objects: List,
    goal_category: str,
    goals: List[ObjectGoal],
    scene_state=None,
    num_episodes: int = -1,
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_target: int = 1000,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        # target_positions = (
        #     pydash.chain()
        #     .map(lambda g: g.view_points)
        #     .flatten().map(lambda v: v.agent_state.position)(goals)
        # )
        # Cache this transformation
        target_positions = np.array(
            list(
                itertools.chain(
                    *(
                        (
                            view_point.agent_state.position
                            for view_point in g.view_points
                        )
                        for g in goals
                    )
                )
            )
        )

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            if (
                source_position is None
                or np.any(np.isnan(source_position))
                or not sim.is_navigable(source_position)
            ):
                raise RuntimeError("Unable to find valid starting location")
            if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
                continue
            compat_outputs = is_compatible_episode(
                source_position,
                target_positions,
                sim,
                goals,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            is_compatible = compat_outputs[0]

            if is_compatible:
                (
                    is_compatible,
                    dist,
                    euclid_dist,
                    source_rotation,
                    geodesic_distances,
                    goals_sorted,
                    shortest_path,
                    ending_state,
                ) = compat_outputs
                shortest_paths = [shortest_path]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=sim.habitat_config.SCENE,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=shortest_paths,
                    scene_state=scene_state,
                    info={
                        "geodesic_distance": dist,
                        "euclidean_distance": euclid_dist,
                        "closest_goal_object_id": goals_sorted[0].object_id,
                        # "navigation_bounds": sim.pathfinder.get_bounds(),
                        # "best_viewpoint_position": ending_state.position,
                    },
                    goals=goals_sorted,
                )

                episode_count += 1
                yield episode
                break

        if retry == number_retries_per_target:
            raise RuntimeError("Unable to find valid starting location")
