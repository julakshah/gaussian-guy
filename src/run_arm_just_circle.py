#!/usr/bin/env python3

import threading
import cv2
import numpy as np
import shutil
import time
import rospy
import open3d
from arm_controller import arm_controller
from camera_controller import camera_controller
from a_star_octree import go_to_goal, find_unknown_leaves
from octree import OctreeNode
import sys
import os

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'widowx_arm/bridge_data_robot-main/widowx_envs/widowx_envs'))
# # When running in container, also add the container's widowx_envs package location
# if os.path.exists('/home/robonet/widowx_envs'):
#     sys.path.insert(0, '/home/robonet/widowx_envs')


def _add_widowx_envs_to_syspath():
    candidates = []
    candidates.append(os.path.join(os.path.dirname(
        __file__), 'widowx_arm', 'bridge_data_robot-main', 'widowx_envs', 'widowx_envs'))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates.append(os.path.join(repo_root, 'widowx_arm',
                      'bridge_data_robot-main', 'widowx_envs', 'widowx_envs'))
    candidates.append('/home/robonet/widowx_envs')
    candidates.append('/home/robonet/host_src/src')

    for p in candidates:
        if os.path.isdir(p):
            if p not in sys.path:
                sys.path.insert(0, p)
            return


_add_widowx_envs_to_syspath()
from widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus


def main():
    rospy.init_node('main_node', anonymous=True)

    # Initialize Camera before anything else
    camera = camera_controller()
    camera_thread = threading.Thread(target=camera.loop_wrapper)
    camera_thread.daemon = True
    camera_thread.start()

    # Initialize Arm -> Go to initial scan position
    controller = arm_controller()
    while (controller.scan_position_reached != True):
        time.sleep(0.1)

    # Find center point code here

    controller.reset_arm()

    # Add circle path to trajectory list
    controller.circle_path([0.275, 0])

    # Start movement based on trajectory list
    arm_thread = threading.Thread(target=controller.loop_wrapper)
    arm_thread.start()

    positions = []
    frames = []

    # While full rotation isn't complete, waits for each point to be reached
    #   stores the position and image at that point and tells arm to continue
    while (controller.has_objective):
        if (controller.reached_objective):
            controller.destinations_reached += 1
            print(f"Objective {controller.destinations_reached} reached")
            time.sleep(1)

            image = camera.get_rgb()
            image_depth = camera.get_depth()
            pos = controller.last_objective

            positions.append(pos)
            frames.append(image)

            time.sleep(1)
            controller.next()

    controller.reset_arm()
    # Go to reset position (0.3, 0.0, 0.15, 0, 0, 0)

    # Save frames in folder. Simpler: use `../images` (one level up from src).
    frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    # remove existing and recreate
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(len(frames)):
        pos = positions[i]

        # Files name example: image1_(0.2,0.05,0.1,0,0.3,0)
        file_name_cv2 = f"image{i}_({pos[0]},{pos[1]},{pos[2]},{pos[3]},{pos[4]},{pos[5]}).jpg"
        save_path_cv2 = os.path.join(frames_dir, file_name_cv2)

        cv2.imwrite(save_path_cv2, frames[i])
        print(f"Saved image {i} to: {save_path_cv2}")

    print(f"All images saved to: {frames_dir}")

    # shutdown
    controller.shutdown()


if __name__ == "__main__":
    main()
