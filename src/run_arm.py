#!/usr/bin/env python3

import threading
import cv2
import numpy as np
import shutil
import time
import rospy
import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # REQUIRED in 3.8.x
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
# COPILOT STOP MOVING THIS IMPORT TO THE TOP IT DOESN'T WORK AT THE TOP AND NEVER HAS
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

    # Create Octree and hardcode intrinsic values of Realsense camera
    intrinsics = open3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=587.85443484,
        fy=576.39067123,
        cx=320.5050574967,
        cy=230.11271642
    )
    octree = OctreeNode(position=(0.3, 0, 0.15), r=0.15, min_r=0.01)
    unknowns_remain = True

    # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # octree.draw(ax)
    # ax.set_xlim(-25,25)
    # ax.set_ylim(-25,25)
    # ax.set_zlim(-25,25)
    # ax.set_box_aspect([1,1,1])
    # plt.show()

    # Add circle path to trajectory list
    controller.circle_path([0.275, 0])
    time.sleep(2)

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

            #### ---- Beginning of Octree Stuff ---- ####
            # Update Octree during circle path
            color = np.asarray(image, dtype=np.uint8)
            color = np.ascontiguousarray(color)

            depth = np.asarray(image_depth, dtype=np.uint16)
            depth = np.ascontiguousarray(depth)

            color_o3d = open3d.geometry.Image(color)
            depth_o3d = open3d.geometry.Image(depth)

            rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1000.0,   # uint16 = millimeters
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )


            point_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsics)  # does extrinsics send it to world frame?
            camera_pose = pose_to_transform(pos)
            point_cloud.transform(camera_pose)
            down_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
            start_pos = pos[0:3]

            for point in down_point_cloud.points:
                tupled_point = tuple(point)
                # filter out table points
                if (tupled_point[0] < 0.13) or (tupled_point[0] > 0.43) or (tupled_point[1] < -0.2) or (tupled_point[1] > 0.2) or (tupled_point[2] < 0.01) or (tupled_point[2] > 0.3):
                    continue

                octree.insert_obstacle(tupled_point)
                # raycast to update with empties
                octree.raycast(start_pos, tupled_point)

            # ax.cla
            # octree.draw(ax)
            #### ---- END of Octree Stuff ---- ####

            time.sleep(1)
            controller.next()

    # Go to reset position (0.3, 0.0, 0.15, 0, 0, 0)
    controller.reset_arm()
    controller.last_objective = np.array([0.3, 0.0, 0.15, 0, 0, 0])

    while unknowns_remain:

        image = camera.get_rgb()
        image_depth = camera.get_depth()
        pos = controller.last_objective

        color_o3d = open3d.geometry.Image(image)
        # Prepare depth for Open3D (2D, supported dtype)
        depth_o3d = open3d.geometry.Image(image_depth)

        # Create RGBD image      --------------------------------------------------------------------- COULD BE BROKE
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1000.0,     # for uint16 depth in millimeters
            depth_trunc=3.0,        # meters, optional
            convert_rgb_to_intensity=False
        )

        point_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics)  # does extrinsics send it to world frame?
        camera_pose = pose_to_transform(pos)

        point_cloud.transform(camera_pose)
        down_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
        start_pos = pos[0:3]

        for point in down_point_cloud.points:
            tupled_point = tuple(point)
            # filter out table points
            if (tupled_point[0] < 0.13) or (tupled_point[0] > 0.43) or (tupled_point[1] < -0.2) or (tupled_point[1] > 0.2) or (tupled_point[2] < 0.01) or (tupled_point[2] > 0.3):
                continue

            print(tupled_point)

            octree.insert_obstacle(tupled_point)
            # raycast to update with empties
            octree.raycast(start_pos, tupled_point)

        path = go_to_goal(octree, start_pos)
        for waypoint in path:
            # `waypoint` is an OctreeNode; use its position tuple and build a 6-DOF pose
            pos = np.array(waypoint.position)

            # Find rpw towards 0.275,0,0 from pos
            roll = np.arctan(pos[1]/(pos[0]-0.275)) - np.pi
            pitch = np.arctan((pos[0]-0.275)/pos[2]) + 1.5

            pose = np.concatenate([pos, np.array([roll, pitch, 0.0])])


            controller.add_to_trajectory(pose)

        while (controller.has_objective):
            if (controller.reached_objective):
                controller.next()
            time.sleep(0.1)

        controller.destinations_reached += 1
        print(f"Objective {controller.destinations_reached} reached")

        image = camera.get_rgb()
        pos = controller.last_objective

        positions.append(pos)
        frames.append(image)

        unknown_leaves = []
        find_unknown_leaves(octree, unknown_leaves)
        if not unknown_leaves:
            unknowns_remain = False

    print("Object scanned")

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    octree.draw(ax)

    limit = octree.r * 1.1
    ax.set_xlim(-limit + 0.275, limit + 0.275)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    plt.show()

    controller.shutdown()

def pose_to_transform(pos):
    cr, sr = np.cos(pos[3]), np.sin(pos[3])
    cp, sp = np.cos(pos[4]), np.sin(pos[4])
    cy, sy = np.cos(pos[5]), np.sin(pos[5])

    R_x = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    R_y = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])

    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])

    R = R_z @ R_y @ R_x

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pos[0], pos[1], pos[2]]

    return T


    # Save frames in folder. Simpler: use `../images` (one level up from src).
    # frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    # # remove existing and recreate
    # if os.path.isdir(frames_dir):
    #     shutil.rmtree(frames_dir)
    # os.makedirs(frames_dir, exist_ok=True)

    # for i in range(len(frames)):
    #     pos = positions[i]

    #     # Files name example: image1_(0.2,0.05,0.1,0,0.3,0)
    #     file_name_cv2 = f"image{i}_({pos[0]},{pos[1]},{pos[2]},{pos[3]},{pos[4]},{pos[5]}).jpg"
    #     save_path_cv2 = os.path.join(frames_dir, file_name_cv2)

    #     cv2.imwrite(save_path_cv2, frames[i])
    #     print(f"Saved image {i} to: {save_path_cv2}")

    # print(f"All images saved to: {frames_dir}")

    # shutdown


if __name__ == "__main__":
    main()
