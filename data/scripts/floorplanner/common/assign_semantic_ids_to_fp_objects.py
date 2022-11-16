import json
import os

from tqdm import tqdm

# path containing individual .txt files containing list of object IDs for each semantic category
semantics_path = "data/scene_datasets/floorplanner/v1/configs/semantics"
objects_path = "data/scene_datasets/floorplanner/v1/configs/objects"

semantic_categories = {
    "misc": 0,
    "bed": 1,
    "chair": 2,
    "potted_plant": 3,
    "sofa": 4,
    "toilet": 5,
    "tv": 6,
}

## create semantic dict
if __name__ == "__main__":
    semantic_dict = {}

    for cat in semantic_categories.keys():
        if cat == "misc":
            continue
        file_path = os.path.join(semantics_path, cat + ".txt")

        assert os.path.exists(file_path), file_path

        cat = cat.split(".")[0]

        with open(file_path, "r") as f:
            cat_ids = f.readlines()[1:]
            semantic_dict[cat] = []

            for id in cat_ids:
                semantic_dict[cat].append(id.strip())

    print("Number of objects in each category:")
    for key in semantic_dict.keys():
        print(key, len(semantic_dict[key]))

    print("Assigning semantic IDs:")
    objects = [x for x in os.listdir(objects_path) if x.endswith(".json")]
    misc_ctr = 0
    for obj in tqdm(objects):
        obj_path = os.path.join(objects_path, obj)

        obj = obj.split(".")[0]

        obj_cat = None

        for key in semantic_dict.keys():
            if obj in semantic_dict[key]:
                obj_cat = semantic_categories[key]
                break

        if obj_cat is None:
            obj_cat = semantic_categories["misc"]
            misc_ctr += 1

        with open(obj_path, "r") as f:

            obj_json = json.load(f)
            if "semantic_id" in obj_json.keys():
                print(
                    f"{obj_path} already has semantic ID assigned. Skipping."
                )
                continue
            obj_json["semantic_id"] = obj_cat

        with open(obj_path, "w") as f:
            json.dump(obj_json, f)

    print("Number of non-miscellaneous objects:", len(objects) - misc_ctr)
