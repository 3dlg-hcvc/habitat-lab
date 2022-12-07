import json
import os

import yaml
from tqdm import tqdm

# path containing individual .txt files containing list of object IDs for each semantic category
semantics_path = "data/scene_datasets/floorplanner/v1/configs/semantics"
goal_categories_path = (
    "data/scene_datasets/floorplanner/v1/goal_categories.yaml"
)
objects_path = "/nethome/mkhanna37/flash1/test/scene-builder-datasets/fphab/habitat-lab/data/scene_datasets/floorplanner/v1/configs/objects"

# alarm_clock 0
# bathtub 1
# bed 2
# book 3
# bottle 4
# bowl 5
# cabinet 6
# carpet 7
# chair 8
# chest_of_drawers 9
# couch 10
# cushion 11
# drinkware 12
# fireplace 13
# fridge 14
# laptop 15
# oven 16
# picture 17
# plate 18
# potted_plant 19
# shelves 20
# shoes 21
# shower 22
# sink 23
# stool 24
# table 25
# table_lamp 26
# toaster 27
# toilet 28
# tv 29
# vase 30
# wardrobe 31
# washer_dryer 32

if __name__ == "__main__":
    semantic_dict = {}

    ## create semantic dict
    with open(goal_categories_path, "r") as f:
        goal_categories = yaml.safe_load(f)

    MISC_CATEGORY_ID = len(goal_categories)

    for cat in goal_categories:
        file_path = os.path.join(semantics_path, cat + ".yaml")
        assert os.path.exists(file_path), file_path

        with open(file_path, "r") as f:
            cat_ids = yaml.safe_load(f)
            semantic_dict[cat] = cat_ids

    print("Number of objects in each category:")
    for key in semantic_dict.keys():
        print(key, len(semantic_dict[key]))

    print("-" * 40)

    print("Category semantic ID mapping:")
    for key in semantic_dict.keys():
        print(key, goal_categories.index(key))

    ## assign semantic IDs
    print("Assigning semantic IDs:")
    objects = [x for x in os.listdir(objects_path) if x.endswith(".json")]
    misc_ctr = 0

    for obj in tqdm(objects):
        obj_path = os.path.join(objects_path, obj)
        obj = obj.split(".")[0]
        obj_cat = None

        for key in semantic_dict.keys():
            if obj in semantic_dict[key]:
                obj_cat = goal_categories.index(key)
                break

        if obj_cat is None:
            obj_cat = MISC_CATEGORY_ID
            misc_ctr += 1

        with open(obj_path, "r") as f:
            obj_json = json.load(f)
            obj_json["semantic_id"] = obj_cat

        with open(obj_path, "w") as f:
            json.dump(obj_json, f)

    print("Number of non-miscellaneous objects:", len(objects) - misc_ctr)
