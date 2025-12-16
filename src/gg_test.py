#!/usr/bin/env python3

import threading
import cv2
import numpy as np
import os
import shutil
import time
import rospy
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from gg_arm_controller import arm_controller
from gg_camera_controller import camera_controller
from gg_a_star_octree import go_to_goal, find_unknown_leaves
from gg_octree import OctreeNode
import open3d



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

    print("Scan Position Reached (controller)")

    time.sleep(1)
    # Find center point code here

    # Add circle path to trajectory list
    controller.reset_arm()
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
            time.sleep(0.25)
            positions.append(controller.last_objective)
            frames.append(camera.get_rgb())
            time.sleep(0.25)
            controller.next()

    # Go to reset position (0.3, 0.0, 0.15, 0, 0, 0)
    controller.reset_arm()

    octree = OctreeNode(position=(0.3,0,0.15), r=0.15, min_r=0.01)      
    unknowns_remain = True

    while unknowns_remain:

        # -> call camera API to get image

        point_cloud = open3d.geometry.create_point_cloud_from_rgbd_image(image, intrinsics, extrinsic) # does extrinsics send it to world frame?
        camera_pose = controller.last_objective
        point_cloud.transform(camera_pose)
        down_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
        start_pos = camera_pose[0:3]

        for point in down_point_cloud.points:
            tupled_point = tuple(point)
            # filter out table points
            if (tupled_point[0] < 0.13) or (tupled_point[0] > 0.43) or (tupled_point[1] < -0.2) or (tupled_point[1] > 0.2) or (tupled_point[2] < 0.01) or (tupled_point[2] > 0.3):
                continue
            
            octree.insert_obstacle(tupled_point)
            ## raycast to update with empties
            octree.raycast(start_pos, tupled_point)

        path = go_to_goal(octree, start_pos)
        for waypoint in path:
            controller.add_to_trajectory(np.concatenate(waypoint,[0,1.5,0]))

        while(controller.has_objective()):
            time.sleep(0.1)

        # get camera frame
        
        unknown_leaves = []
        find_unknown_leaves(octree, unknown_leaves)
        if not unknown_leaves:
            unknowns_remain = False

    print("Object scanned")

    # Save frames in folder
    frames_dir = os.path.join(os.path.dirname(__file__), 'frames')
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
