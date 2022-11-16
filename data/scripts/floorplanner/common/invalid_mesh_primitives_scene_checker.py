import json
import os

from tqdm import tqdm

gltf_scenes_path = "glb-arch-only-jsons"

files = os.listdir(gltf_scenes_path)

faulty_scenes = []

for file in tqdm(files):
    file_path = os.path.join(gltf_scenes_path, file)
    with open(file_path, "r") as f:
        data = json.load(f)

    meshes = data["meshes"]
    found = False
    for mesh in meshes:
        for prim in mesh["primitives"]:
            mode = prim["mode"]
            if mode != 4:
                faulty_scenes.append(file)
                found = True
                break
        if found:
            break

print(len(faulty_scenes), faulty_scenes)
