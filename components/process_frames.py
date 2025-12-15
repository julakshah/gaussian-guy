"""
Script to process the position-stamped image frames recorded by the camera
"""
import pycolmap as pycolmap
import numpy as np
import os
import re
import json
import sys

def main(path: str):
    cwd = os.getcwd()
    src = os.path.join(cwd,path)
    dst = os.path.join(cwd,"colmap/",path)
    pattern = r"\(.*\)" # matches the parenthetical expression in "frameX_(x,y,z,roll,pitch,yaw).jpg"
    frame_locs = {}

    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file():
                name = entry.name
                match1 = re.search(pattern=pattern,string=name)
                cropped = match1.group(0)[1:-1]
                name_split = cropped.split(',')

                # Get our location parameters
                frame_name = int(name[5]) # hardcoded for now, should be changed to search the string
                frame_locs[frame_name] = [name_split]
    
    with open(os.path.join(dst,"frame_locations.json"),'w') as f:
        json.dump(frame_locs,f,indent=4)
    
    # Automate COLMAP
    

if __name__ == "__main__":
    if len(sys.argv < 2):
        print("Please enter a valid path to an image directory")
        sys.exit()
    else:
        path = str(sys.argv[1])
        main(path=path)