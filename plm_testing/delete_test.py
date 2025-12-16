import scipy.io
import os
import numpy as np

# Update this path if needed
base_dir = "/home/connor/Downloads/rgbd-scenes"
mat_path = os.path.join(base_dir, "desk/desk_1.mat")

def inspect_bboxes(mat_path):
    print(f"--- Inspecting {mat_path} ---")
    try:
        mat = scipy.io.loadmat(mat_path)
        if 'bboxes' in mat:
            bbox_data = mat['bboxes']
            # MATLAB structs are loaded as numpy structured arrays
            if hasattr(bbox_data, 'dtype'):
                print(f"Fields found inside 'bboxes': {bbox_data.dtype.names}")
            else:
                print("'bboxes' is not a structured array.")
        else:
            print("'bboxes' key not found.")
    except Exception as e:
        print(f"Error reading MAT: {e}")

def find_pose_files(start_dir):
    print(f"\n--- Searching for .pose or .txt files in {start_dir} ---")
    found = False
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".pose") or file.endswith(".txt"):
                print(f"FOUND: {os.path.join(root, file)}")
                found = True
    if not found:
        print("No .pose or .txt files found.")

if __name__ == "__main__":
    # 1. Dig into the mat file
    inspect_bboxes(mat_path)
    
    # 2. Hunt for the missing pose files
    find_pose_files(base_dir)