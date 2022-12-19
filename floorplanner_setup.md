# FloorPlanner Habitat Experiments

## Table of contents
   1. [Scene dataset setup](#scene-dataset-setup)
   2. [Code structure](#code-structure)
   3. [PointNav](#pointnav)
   4. [ObjectNav](#objectnav)

## Scene dataset setup:
1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim):

    Build [`v0.2.3`](https://github.com/facebookresearch/habitat-sim/tree/v0.2.3) branch from source ([installation instructions](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source)).

    
    **IMPORTANT** ⚠️: Currently, habitat-sim defaults to not including static objects in navmesh. However, with the current scene dataset setup, we are loading (static) scene objects on the fly and therefore want them to be included in the navmesh — for accurate navigation in scenes. To allow this, currently, we need to make the following change to our habitat-sim installation and rebuild it: 

    ```py
    #  habitat-sim/src_python/habitat_sim/simulator.py > L251:
    self.recompute_navmesh(self.pathfinder, needed_settings, include_static_objects=True)
    ```

2. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab):

    Install habitat-lab from this [custom fork](https://github.com/mukulkhanna/habitat-lab/tree/floorplanner) (current repository).

    This fork was off habitat-lab v0.2.2 + includes a couple of additional merged PRs. We are yet to port to hab-lab v0.2.3.

    ```bash
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines  # install habitat_baselines

    export PYTHONPATH=$PYTHONPATH:./  # in your ~/.bashrc, add habitat-lab to PYTHONPATH
    ```

3. Download FloorPlanner dataset:

    ```bash
    mkdir data/scene_datasets/
    wget --no-check-certificate https://aspis.cmpt.sfu.ca/projects/scenebuilder/fphab/v0.1.1/fphab.zip -O data/scene_datasets/fphab.zip
    unzip data/scene_datasets/fphab.zip -d data/scene_datasets/floorplanner
    mv data/scene_datasets/floorplanner/fphab data/scene_datasets/floorplanner/v1 # rename to v1
    ```

4. Run demo scene script:

    ```bash
    python data/scripts/floorplanner/common/fp_scene_demo.py
    ```

    This will save on your disk an agent's egocentric RGB observation at a random navigable position and the topdown map for a sample scene.

    <p align="center">
    <img style="margin:1%" src="https://user-images.githubusercontent.com/24846546/208518017-e5394f91-9f27-4041-95d8-d36810c37745.png" height="200">
    <img style="margin:1%" src="https://user-images.githubusercontent.com/24846546/208518005-e912c9a9-0238-41de-bd45-51858cc81e47.png" height="200">
    </p>


## Code structure

```bash
habitat-lab
└── data
    ├──scene_datasets # to store scene dataset assets, config files, stats, and visualizations
    ├──datasets # to store objectnav and pointnav episode datasets
    └── scripts
        ├── habitat-matterport3d-dataset # scripts from HM3D-sem for computing scene stats
        └── floorplanner
            ├── common # stores general purpose scripts
            ├── objectnav # stores objectnav-related scripts
            ├── pointnav
            └── utils
```
---

## PointNav

1. Generate scene dataset splits.
    ```
    python data/scripts/floorplanner/common/create_fp_dataset_splits.py
    ```

2. Run PointNav episode generator.
    ```
    python data/scripts/floorplanner/pointnav/create_fp_pointnav_dataset.py --split train --viz
    ```
    <p align="center">
    <img src="https://user-images.githubusercontent.com/24846546/208518360-9ba7c517-293a-4170-bd91-0771e70280db.jpeg" width=60%>
    </p>

3. Train DDPPO-based PointNav agent!
    ```
    python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_fp.yaml --run-type train
    ```


## ObjectNav

1. Generate scene dataset splits (if not already done).
    ```
    python data/scripts/floorplanner/common/create_fp_dataset_splits.py
    ```

2. Run ObjectNav episode generator.
    
    To create an episode dataset with 33 goal categories (specified in `semantics` dir of dataset), run:
    ```
    python data/scripts/floorplanner/objectnav/create_fp_objectnav_dataset.py --split train
    ```

    Episode visualizations like the one below are saved in `data/datasets/objectnav/floorplanner/v0.1.1/viz`.

    <p align="center">
    <img src="https://user-images.githubusercontent.com/24846546/208518373-8a4dcf7d-3b75-4b35-b2a5-84fbce65f392.jpeg" width=60%>
    </p>

3. Train RGBD DDPPO-based ObjectNav agent!
    ```
    python -u habitat_baselines/run.py --exp-config habitat_baselines/config/objectnav/ddppo_objectnav_fp.yaml  --run-type train
    ```

    Training logs and checkpoints are saved in `data/training/floorplanner/v0.1.1/`.

    > Note: The default `num_envs` count in the config file was specified for an A40 GPU.