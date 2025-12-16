"""
Script to process the position-stamped image frames recorded by the camera
"""
import pycolmap as pycolmap
import subprocess
from pathlib import Path
import numpy as np
import os
import re
import json
import sys

def run_sfm(image_dir: str, out_dir: str):
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    database_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # feature extraction via colmap
    pycolmap.extract_features(
        database_path,
        image_dir,
        image_options={"single_camera": True}, # one camera is shared
       )

    # Feature mapping via colmap --- exhaustive as we don't know the order of our frames
    pycolmap.match_exhaustive(database_path)

    # Generate SfM data via incremental mapping
    reconstructions = pycolmap.incremental_mapping(database_path,image_dir,sparse_dir,)

    if not reconstructions:
        raise RuntimeError("Failed to create reconstructions")

    # Take the largest reconstruction
    recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    model_dir = sparse_dir / "0"
    model_dir.mkdir(exist_ok=True)
    recon.write(model_dir)  # writes cameras/images/points3D in COLMAP format
    print(f"Wrote sparse model to: {model_dir}")

def main(path: str):
    cwd = os.getcwd()
    src = os.path.join(cwd,path)
    dst = os.path.join(cwd,"colmap/",path)
    pattern = r"\(.*\)" # matches the parenthetical expression in "frameX_(x,y,z,roll,pitch,yaw).jpg"
    frame_locs_train = {}
    frame_locs_test = {}

    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file():
                name = entry.name
                match1 = re.search(pattern=pattern,string=name)
                if match1:
                    cropped = match1.group(0)[1:-1]
                    name_split = cropped.split(',')

                    # Get our location parameters
                    frame_name = int(name[5]) # hardcoded for now, should be changed to search the string
                    if frame_name % 4 == 0: #change this if we want
                        frame_locs_test[frame_name] = name_split
                    else:
                        frame_locs_train[frame_name] = name_split
    
    os.makedirs(dst,exist_ok=True)
    with open(os.path.join(dst,"frame_locations_train.json"),'w') as f:
        json.dump(frame_locs_train,f,indent=4)

    with open(os.path.join(dst,"frame_locations_test.json"),'w') as f:
        json.dump(frame_locs_test,f,indent=4)

    # Automate COLMAP
    run_sfm(path,path)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter a valid path to an image directory")
        sys.exit()
    else:
        path = str(sys.argv[1])
        main(path=path)