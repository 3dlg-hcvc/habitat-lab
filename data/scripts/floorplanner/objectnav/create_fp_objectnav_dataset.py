import argparse
import glob
import gzip
import itertools
import json
import lzma
import multiprocessing
import os
import os.path as osp
import pickle
import traceback
from collections import defaultdict
import copy

import cv2
import GPUtil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import trimesh
import yaml
from objectnav_episode_generator import (
    build_goal,
    generate_objectnav_episode_v2,
)
from sklearn.cluster import AgglomerativeClustering

import habitat
import habitat_sim
from data.scripts.floorplanner.utils.utils import (
    COLOR_PALETTE,
    get_topdown_map,
)
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.datasets.object_nav import object_nav_dataset
from habitat.datasets.pointnav.pointnav_generator import ISLAND_RADIUS_LIMIT
from habitat.tasks.rearrange.utils import get_aabb

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

COMPRESSION = ".gz"
# DATASET = "ai2thor/procthor" # "floorplanner"
DATASET = "hm3d"
SCENE_DATASET_VERSION_ID = "v2"
# EPISODE_DATASET_VERSION_ID = "v0.0.8_6_cat" # "v0.0.8_6_cat"
EPISODE_DATASET_VERSION_ID = "v2_28_cat"
# GOAL_CATEGORIES_FILENAME = "33_goal_categories.yaml"
OBJECT_ON_SAME_FLOOR = True  # [UPDATED]
NUM_EPISODES = 50000
MIN_OBJECT_DISTANCE = 1.0
MAX_OBJECT_DISTANCE = 30.0
INDOOR_CHECK = False

NUM_GPUS = 2#len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 20

if "thor" in DATASET:
    DATASET, SUBDATASET = DATASET.split("/")
    # ISLAND_RADIUS_LIMIT = 1.0 # TODO: fetch from constants/cfg file
    INDOOR_CHECK = False

scenes_root_path = f"data/scene_datasets/{DATASET}/{SCENE_DATASET_VERSION_ID}"
semantic_id_mapping_path = os.path.join(
    scenes_root_path, "configs", "semantics", "object_semantic_id_mapping.json"
)
if "thor" not in DATASET:
    scene_splits_path = os.path.join(
        scenes_root_path, "configs", "scene_splits.yaml"
    )
else:
    scene_splits_path = os.path.join(
        scenes_root_path, "configs", "scene_splits", f"{SUBDATASET}.yaml"
    )

with open(scene_splits_path, "r") as f:
    scene_splits = yaml.safe_load(f)

if "thor" not in DATASET:
    output_dataset_folder = (
        f"data/datasets/objectnav/{DATASET}/{EPISODE_DATASET_VERSION_ID}"
    )
else:
    output_dataset_folder = f"data/datasets/objectnav/{DATASET}/{SUBDATASET}/{EPISODE_DATASET_VERSION_ID}"

episode_dataset_viz_folder = os.path.join(
    output_dataset_folder, "viz-30-val-eps-per-scene", "episodes"
)
goal_distances_viz_folder = os.path.join(
    output_dataset_folder, "viz-30-val-eps-per-scene", "goal_distances"
)
failure_viz_folder = os.path.join(
    output_dataset_folder, "viz-30-val-eps-per-scene", "failure_cases"
)

with open(semantic_id_mapping_path, "r") as f:
    semantic_id_mapping = json.load(f)

fp_hm3d_catmap = pd.read_csv(f'data/scene_datasets/{DATASET}/fp-hm3d-mapping.csv')

FP_HM3D_MAP = defaultdict(list)

for i in range(fp_hm3d_catmap.shape[0]):
    for j in range(1, fp_hm3d_catmap.shape[1]):
        if not pd.isnull(fp_hm3d_catmap.iloc[i, j]):
            FP_HM3D_MAP[fp_hm3d_catmap.iloc[i, 0]].append(fp_hm3d_catmap.iloc[i, j])

HM3D_FP_MAP = {}
for fpobj, hmobj in FP_HM3D_MAP.items():
    for ho in hmobj:
        HM3D_FP_MAP[ho] = fpobj

def get_objnav_config(i, scene):

    # if "thor" not in DATASET:
    TASK_CFG = "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d_with_semantic.yaml"
    SCENE_DATASET_CFG = os.path.join(
        scenes_root_path, "hm3d_annotated_basis.scene_dataset_config.json"
    )
    # else:
    #     TASK_CFG = f"habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_{SUBDATASET}_with_semantic.yaml"
    #     SCENE_DATASET_CFG = os.path.join(
    #         scenes_root_path, "ai2thor.scene_dataset_config.json"
    #     )

    objnav_config = get_config(TASK_CFG)

    deviceIds = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
    )

    if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
        deviceId = i % NUM_GPUS
    else:
        deviceId = deviceIds[0]

    with read_write(objnav_config):

        FOV = 90
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            FOV
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            FOV
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = (
            FOV
        )
        # TODO: confirm the width and height
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            320
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            240
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            320
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            240
        )

        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = deviceId
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG
        objnav_config.habitat.simulator.habitat_sim_v0.enable_physics = True

        objnav_config.habitat.task.measurements = {}

    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim(
        "Sim-v0", config=objnav_config.habitat.simulator
    )
    # no need to recompute navmesh because forked hab-sim installation includes static objs by default
    # if INDOOR_CHECK:
    #     sim.compute_navmesh_island_classifications()
    #     for island_idx in sim.indoor_islands:
    #         if sim.pathfinder.island_area(island_idx) > ISLAND_RADIUS_LIMIT:
    #             return sim
    #     return None  # scene has no valid indoor islands
    # else:
    return sim


def dense_sampling_trimesh(triangles, density=25.0, max_points=200000):
    # Create trimesh mesh from triangles
    t_vertices = triangles.reshape(-1, 3)
    t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
    t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces)
    surface_area = t_mesh.area
    n_points = min(int(surface_area * density), max_points)
    t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
    return t_pts

def get_gravity_mobb(object_obb: habitat_sim.geo.OBB):
    bounding_area = [
        (object_obb.local_to_world @ np.array([x, y, z, 1]))[:-1]
        for x, y, z in itertools.product(*([[-1, 1]] * 3))
    ]
    bounding_area = np.array(bounding_area, dtype=np.float32)
    # print('Bounding Area: %s' % bounding_area)
    # TODO Maybe Cache this
    return habitat_sim.geo.compute_gravity_aligned_MOBB(
        habitat_sim.geo.GRAVITY, bounding_area
    )

def get_scene_key(glb_path):
    return osp.basename(glb_path).split(".")[0]


def get_file_opener(fname):
    ext = os.path.splitext(fname)[-1]

    if ext == ".gz":
        file_opener = gzip.open
    elif ext == ".xz":
        file_opener = lzma.open
    else:
        print(ext)
        assert False
    return file_opener


def save_dataset(dset: habitat.Dataset, fname: str):
    file_opener = get_file_opener(fname)
    if (
        os.path.basename(os.path.dirname(fname)) == "content"
        and len(dset.episodes) == 0
    ):
        print("WARNING UNEXPECTED EMPTY EPISODES: %s" % fname)
        return
    with file_opener(fname, "wt") as f:
        if len(dset.episodes) == 0:
            print("WARNING EMPTY EPISODES: %s" % fname)
            f.write(
                json.dumps(
                    {
                        "episodes": [],
                        "category_to_task_category_id": dset.category_to_scene_annotation_category_id,
                        "category_to_scene_annotation_category_id": dset.category_to_scene_annotation_category_id,
                    }
                )
            )
        else:
            dset_str = dset.to_json()
            f.write(dset_str)


def generate_scene(args):
    i, scene, split = args
    objnav_config = get_objnav_config(i, scene)

    sim = get_simulator(objnav_config)
    if sim is None:
        print(f"Scene {scene} has no valid indoor islands.")
        return scene, 0, defaultdict(list), None, 0

    total_objects = total_objects = len(sim.semantic_annotations().objects)

    # Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if total_objects == 0 or not sim.is_navigable(np.array(test_point)):
        print(f"Scene {scene} has {total_objects} objects / is not navigable: %s" % scene)
        sim.close()
        return scene, total_objects, defaultdict(list), None, 0

    objects = []

    # rgm = sim.get_rigid_object_manager()
    # obj_ids = [int(x.split(",")[5]) for x in rgm.get_objects_info()[1:]]

    # # recording (bboxes, category, etc.) info about all objects in scene
    # for obj_id in tqdm.tqdm(
    #     obj_ids,
    #     desc="Generating object data",
    # ):
    #     source_obj = rgm.get_object_by_id(obj_id)
    #     semantic_id = source_obj.semantic_id

    #     category_name = None
    #     for cat, cat_id in semantic_id_mapping.items():
    #         if cat_id == semantic_id:
    #             category_name = cat
    #             break

    #     # CRUCIAL: replacing semantic id with object id to fetch instance maps
    #     for node in source_obj.visual_scene_nodes:
    #         node.semantic_id = obj_id

    #     if (
    #         category_name is None or category_name not in goal_categories
    #     ):  # non-goal category
    #         continue

    #     if source_obj is None:
    #         print("=====> Source object is None. Skipping.")
    #         continue

    #     aabb = get_aabb(obj_id, sim, transformed=True)
    #     center = np.array(source_obj.translation)
    #     sizes = np.array(source_obj.root_scene_node.cumulative_bb.size())
    #     rotation = source_obj.rotation

    #     obb = habitat_sim.geo.OBB(center, sizes, rotation)
    #     obj = {
    #         "center": source_obj.translation,
    #         "id": obj_id,
    #         "object_name": rgm.get_object_handle_by_id(obj_id),
    #         "obb": obb,
    #         "aabb": aabb,
    #         "category_id": semantic_id,
    #         "category_name": category_name,
    #     }
    #     objects.append(obj)

    # print("Scene loaded.")

        ####### documenting info about all objects (bboxes, category info, and such)

    for source_id, source_obj in enumerate(
        tqdm.tqdm(
            sim.semantic_annotations().objects,
            desc="Generating object data",
        )
    ):
        if source_obj is None:
            print('=====> Source object is None. Skipping.')
            continue
        raw_name = copy.deepcopy(source_obj.category.name(""))
        ########################################################################
        # # Apply annotation correction if available
        # obj_id = int(source_obj.id.split("_")[-1])
        # scene_name = scene.split("/")[-2]
        # if (scene_name, obj_id) in ANNOTATION_CORRECTIONS:
        #     expec_raw_name = ANNOTATION_CORRECTIONS[(scene_name, obj_id)][1]
        #     if raw_name != expec_raw_name:
        #         print(f"Raw name: {raw_name}, Annot corr: {ANNOTATION_CORRECTIONS[(scene_name, obj_id)]}")
        #     assert raw_name == expec_raw_name
        #     raw_name = ANNOTATION_CORRECTIONS[(scene_name, obj_id)][1]
        ########################################################################
        raw_name = raw_name.strip().lower()
        if raw_name not in HM3D_FP_MAP: #non-goal category
            continue
        category_name = HM3D_FP_MAP[raw_name] #HM3D_RAW_TO_CAT_MAPPING[raw_name]
        category_id = semantic_id_mapping[category_name] #source_obj.category.index("")

        # if category_name not in wordlist:
        #     continue
        if np.all(source_obj.obb.sizes == 0):
            continue
        if category_name == None:
            print("ERROR NONE CATEGORY NAME: %s %d" % (scene, source_id))
            continue

        obj = {
            "center": source_obj.aabb.center,
            "id": int(source_obj.id.split("_")[-1]),
            "object_name": source_obj.id,
            "obb": source_obj.obb,
            "aabb": source_obj.aabb,
            "gravity_mobb": get_gravity_mobb(source_obj.obb),
            "category_id": category_id,
            "category_name": category_name,
        }
        objects.append(obj)

    print("Scene loaded.")
    fname_obj = (
        f"{output_dataset_folder}/{split}/scene_goals/{scene}_goal_objs.pkl"
    )
    fname = (
        f"{output_dataset_folder}/{split}/content/{scene}.json{COMPRESSION}"
    )

    ############################################################################
    # Pre-compute goals
    ############################################################################
    obj_file_exists = os.path.exists(fname_obj)

    if obj_file_exists:
        with open(fname_obj, "rb") as f:
            goals_by_category = pickle.load(f)
        total_objects_by_cat = {
            k: len(v["goals"]) for k, v in goals_by_category.items()
        }
    else:
        # goals_by_category = defaultdict(list)
        goals_by_category = {}
        cell_size = (
            objnav_config.habitat.simulator.agents.main_agent.radius / 2.0
        )
        categories_to_counts = {}

        for obj in tqdm.tqdm(objects, desc="Objects for %s:" % scene):
            # print(f'Object id: {obj["id"]}, ({obj["category_name"]})')
            if obj["category_name"] not in categories_to_counts:
                categories_to_counts[obj["category_name"]] = [0, 0]
            categories_to_counts[obj["category_name"]][1] += 1
            print(
                obj["category_name"], obj["category_id"], obj["category_name"]
            )

            goal, topdown_map = build_goal(
                sim,
                object_id=obj["id"],
                object_name_id=obj["object_name"],
                object_category_name=obj["category_name"],
                object_category_id=obj["category_id"],
                object_position=obj["center"],
                object_aabb=obj["aabb"],
                object_obb=obj["obb"],
                cell_size=cell_size,
                grid_radius=3.0,
                indoor_check=INDOOR_CHECK,
            )

            if goal == None:
                os.makedirs(
                    os.path.join(failure_viz_folder, scene), exist_ok=True
                )
                fail_case_path = os.path.join(
                    failure_viz_folder,
                    scene,
                    f'{obj["category_name"]}_{obj["object_name"]}.jpg',
                )
                cv2.imwrite(fail_case_path, topdown_map[:, :, ::-1])
                continue

            categories_to_counts[obj["category_name"]][0] += 1
            if (
                osp.basename(scene) + "_" + obj["category_name"]
                not in goals_by_category.keys()
            ):
                goals_by_category[
                    osp.basename(scene) + "_" + obj["category_name"]
                ] = {"goals": [], "topdown_maps": {}}

            goals_by_category[
                osp.basename(scene) + "_" + obj["category_name"]
            ]["goals"].append(goal)
            goals_by_category[
                osp.basename(scene) + "_" + obj["category_name"]
            ]["topdown_maps"][obj["id"]] = topdown_map

        for obj_cat in sorted(list(categories_to_counts.keys())):
            nvalid, ntotal = categories_to_counts[obj_cat]
            print(
                f"Category: {obj_cat:<15s} | {nvalid:03d}/{ntotal:03d} instances"
            )
        os.makedirs(osp.dirname(fname_obj), exist_ok=True)
        total_objects_by_cat = {
            k: len(v["goals"]) for k, v in goals_by_category.items()
        }
        with open(fname_obj, "wb") as f:
            pickle.dump(goals_by_category, f)

    ############################################################################
    # Cluster points on the navmesh
    ############################################################################

    obj_save_path = os.path.join(
        output_dataset_folder, split, "scene_goals", f"{scene}_clusters.pkl"
    )
    if os.path.isfile(obj_save_path):
        with open(obj_save_path, "rb") as fp:
            obj_data = pickle.load(fp)
            cluster_infos = obj_data["cluster_infos"]
            goal_category_to_cluster_distances = obj_data[
                "goal_category_to_cluster_distances"
            ]
    else:
        # Discover navmesh clusters
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_pc = dense_sampling_trimesh(navmesh_triangles)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="euclidean",
            distance_threshold=1.0,
        ).fit(navmesh_pc)
        labels = clustering.labels_
        n_clusters = clustering.n_clusters_
        cluster_infos = []
        for i in range(n_clusters):
            center = navmesh_pc[labels == i, :].mean(axis=0)
            if sim.pathfinder.is_navigable(center):
                if sim.island_radius(center) < ISLAND_RADIUS_LIMIT:
                    continue
                if INDOOR_CHECK:
                    if (
                        sim.pathfinder.get_island(center)
                        not in sim.indoor_islands
                    ):
                        continue
                center = np.array(sim.pathfinder.snap_point(center)).tolist()
                locs = navmesh_pc[labels == i, :].tolist()
                stddev = np.linalg.norm(np.std(locs, axis=0)).item()
                cluster_infos.append(
                    {"center": center, "locs": locs, "stddev": stddev}
                )

        print(f"====> Calculated cluster infos. # clusters: {n_clusters}")

        # Calculate distances from goals to cluster centers
        goal_category_to_cluster_distances = {}
        for cat, data in goals_by_category.items():
            object_vps = []
            for inst_data in data["goals"]:
                for view_point in inst_data.view_points:
                    object_vps.append(view_point.agent_state.position)
            goal_distances = []
            for i, cluster_info in enumerate(cluster_infos):
                from data.scripts.floorplanner.utils.utils import (
                    get_topdown_map,
                )

                if i == 0:
                    td = get_topdown_map(sim, center, marker=None)
                dist = sim.geodesic_distance(
                    cluster_info["center"], object_vps
                )

                goal_distances.append(dist)
            goal_category_to_cluster_distances[cat] = goal_distances

        with open(obj_save_path, "wb") as fp:
            pickle.dump(
                {
                    "cluster_infos": cluster_infos,
                    "goal_category_to_cluster_distances": goal_category_to_cluster_distances,
                },
                fp,
            )

    ############################################################################
    # Compute ObjectNav episodes
    ############################################################################
    if os.path.exists(fname):
        print("Scene already generated. Skipping")
        sim.close(destroy=True)
        return scene, total_objects, total_objects_by_cat, None, None

    total_valid_cats = len(total_objects_by_cat)
    dset = habitat.datasets.make_dataset("ObjectNav-v1")

    dset.category_to_task_category_id = semantic_id_mapping
    dset.category_to_scene_annotation_category_id = semantic_id_mapping
    dset_goals_by_category = {
        k: {"goals": v["goals"]} for k, v in goals_by_category.items()
    }
    # dset_goals_by_category = {
    #     k: {v["goals"]} for k, v in goals_by_category.items()
    # }
    dset.goals_by_category = dset_goals_by_category
    scene_dataset_config = objnav_config.habitat.simulator.scene_dataset

    with tqdm.tqdm(total=NUM_EPISODES, desc=scene) as pbar:

        eps_generated = 0
        for goal_cat, goals in goals_by_category.items():
            eps_per_obj = int(
                NUM_EPISODES / total_valid_cats
            )  # equal episodes for each category
            try:
                from data.scripts.floorplanner.utils.utils import (
                    get_topdown_map,
                )

                for i, ep in enumerate(
                    generate_objectnav_episode_v2(
                        sim,
                        goals["goals"],
                        cluster_infos,
                        np.array(goal_category_to_cluster_distances[goal_cat]),
                        num_episodes=eps_per_obj,
                        closest_dist_limit=MIN_OBJECT_DISTANCE,
                        furthest_dist_limit=MAX_OBJECT_DISTANCE,
                        scene_dataset_config=scene_dataset_config,
                        same_floor_flag=OBJECT_ON_SAME_FLOOR,
                        eps_generated=eps_generated,
                        indoor_check=INDOOR_CHECK,
                    )
                ):
                    if "distances" not in goals.keys():
                        goals["distances"] = {}

                    dset.episodes.append(ep)
                    pbar.update()
                    eps_generated += 1
                    goal_obj_id = ep.info["closest_goal_object_id"]

                    # keeping track of distances from each goal object (for plotting)
                    if goal_obj_id not in goals["distances"].keys():
                        goals["distances"][goal_obj_id] = [
                            ep.info["geodesic_distance"]
                        ]
                    else:
                        goals["distances"][goal_obj_id].append(
                            ep.info["geodesic_distance"]
                        )

                    # visualizing start position for each episode
                    goals["topdown_maps"][goal_obj_id] = get_topdown_map(
                        sim,
                        start_pos=ep.start_position,
                        marker="circle",
                        color=COLOR_PALETTE["orange"],
                        radius=3,
                        topdown_map=goals["topdown_maps"][goal_obj_id],
                    )

            except RuntimeError:
                traceback.print_exc()
                obj_cat = goals["goals"][0].object_name
                print(f"Skipping category {goal_cat}; ID: {obj_cat}")
                pbar.update(eps_per_obj - eps_generated)
                continue

            for obj_id, distances in goals["distances"].items():
                obj = [x for x in goals["goals"] if x.object_id == obj_id][0]
                obj_name = obj.object_name
                obj_cat = obj.object_category
                episodes_viz_output_path = os.path.join(
                    episode_dataset_viz_folder, scene
                )
                os.makedirs(episodes_viz_output_path, exist_ok=True)
                episode_viz_output_filename = os.path.join(
                    episodes_viz_output_path, f"{obj_cat}_{obj_name}.jpg"
                )
                tdm = goals["topdown_maps"][obj_id]
                cv2.imwrite(episode_viz_output_filename, tdm[:, :, ::-1])

                goal_distances_viz_output_path = os.path.join(
                    goal_distances_viz_folder, scene
                )
                os.makedirs(goal_distances_viz_output_path, exist_ok=True)
                goal_viz_output_filename = os.path.join(
                    goal_distances_viz_output_path, f"{obj_cat}_{obj_name}.jpg".replace('/', '')
                )
                plt.figure(figsize=(8, 8))

                sns.histplot(data=distances)
                plt.title(f"{obj_cat}_{obj_name}", fontsize=15)
                plt.xlabel("distance to goal")
                plt.tight_layout()
                plt.savefig(goal_viz_output_filename)
                plt.close()

    os.makedirs(osp.dirname(fname), exist_ok=True)
    save_dataset(dset, fname)
    sim.close(destroy=True)
    return (
        scene,
        total_objects,
        total_objects_by_cat,
        fname,
        len(dset.episodes),
    )


def read_dset(json_fname):
    dset2 = habitat.datasets.make_dataset("ObjectNav-v1")
    file_opener = get_file_opener(json_fname)

    with file_opener(json_fname, "rt") as f:
        dset2.from_json(f.read())
    return dset2


def prepare_inputs(split):
    scenes = scene_splits[split]
    return [(i, scene, split) for i, scene in enumerate(scenes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "*"],
        required=True,
        type=str,
    )
    args = parser.parse_args()

    mp_ctx = multiprocessing.get_context("forkserver")

    np.random.seed(1234)
    if args.split == "*":
        inputs = []
        for split in ["train", "val", "test"]:
            inputs += prepare_inputs(split)
    else:
        inputs = prepare_inputs(args.split)

    GPU_THREADS = NUM_GPUS * TASKS_PER_GPU
    print("GPU threads:", GPU_THREADS)
    print("*" * 50)

    # # [DEBUG]:
    # total_all = 0
    # subtotals = []
    # for inp in tqdm.tqdm(inputs):
    #     if inp[1] != 'ProcTHOR-Train-0':
    #         continue
    #     scene, subtotal, subtotal_by_cat, fname = generate_scene(inp)
    #     total_all += subtotal
    #     subtotals.append(subtotal_by_cat)

    # Generate episodes for all scenes
    os.makedirs(output_dataset_folder, exist_ok=True)
    no_episode_scenes = []

    # Create split outer files
    if args.split == "*":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]
    for split in splits:
        dset = habitat.datasets.make_dataset("ObjectNav-v1")
        dset.category_to_task_category_id = semantic_id_mapping
        dset.category_to_scene_annotation_category_id = semantic_id_mapping
        global_dset = (
            f"{output_dataset_folder}/{split}/{split}.json{COMPRESSION}"
        )
        if os.path.exists(global_dset):
            os.remove(global_dset)
        if not os.path.exists(os.path.dirname(global_dset)):
            os.mkdir(os.path.dirname(global_dset))
        jsons_gz = glob.glob(
            f"{output_dataset_folder}/{split}/content/*.json{COMPRESSION}"
        )

        save_dataset(dset, global_dset)

    with mp_ctx.Pool(GPU_THREADS, maxtasksperchild=2) as pool, tqdm.tqdm(
        total=len(inputs)
    ) as pbar, open(
        os.path.join(output_dataset_folder, "train_subtotals.json"), "w"
    ) as f:
        total_all = 0
        subtotals = []
        for (
            scene,
            subtotal,
            subtotal_by_cat,
            fname,
            num_episodes,
        ) in pool.imap_unordered(generate_scene, inputs):
            pbar.update()
            total_all += subtotal
            subtotals.append(subtotal_by_cat)
            if num_episodes == 0:
                no_episode_scenes.append(scene)
            print("Scene with no episodes:", no_episode_scenes)
        print(total_all)
        print(subtotals)

        json.dump({"total_objects:": total_all, "subtotal": subtotals}, f)
