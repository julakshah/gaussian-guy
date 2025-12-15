#!/usr/bin/env python3

import threading
import cv2
import numpy as np
import os
import math
import shutil
import time
import rospy
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from arm_controller import arm_controller
from camera_controller import camera_controller


def main():
    rospy.init_node('test_node', anonymous=True)

    camera = camera_controller()

    camera_thread = threading.Thread(target=camera.loop_wrapper)
    camera_thread.daemon = True
    camera_thread.start()

    controller = arm_controller()

    while (controller.scan_position_reached != True):
        print("while loop")
        time.sleep(0.5)

    print("Scan Position Reached (controller)")

    time.sleep(5)

    controller.reset_arm()
    controller.circle_path([0.275, 0])

    arm_thread = threading.Thread(target=controller.loop_wrapper)
    arm_thread.start()

    positions = []
    frames = []

    while (controller.has_objective):
        try:
            if (controller.reached_objective):
                controller.destinations_reached += 1
                print(f"Objective {controller.destinations_reached} reached")
                time.sleep(0.25)
                positions.append(controller.last_objective)
                frames.append(camera.get_rgb())
                time.sleep(0.25)
                controller.next()
        except KeyboardInterrupt:
            print("\nShutting down...")
            break

    # A star shit

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
    # rospy.signal_shutdown()

# def find_pickup_droop(pickup_point):
#     a = distance_from_zero_zero(pickup_point)
#     droop = -2.35E-03 + 0.0522*a + -0.216*a**2
#     return droop


# def find_placing_droop(pickup_point, place_point):
#     pickup_disp = distance_from_zero_zero(pickup_point)
#     place_disp = distance_from_zero_zero(place_point)
#     xy_increase = place_disp - pickup_disp
#     return (pickup_disp - max(place_disp, 0.2)) * -0.0529


# def distance_from_zero_zero(point):
#     """
#     Calculate the distance between [0, 0] and another point [x, y2.

#     Args:
#         point (list or tuple): Coordinates of the point [x, y].

#     Returns:
#         float: Distance between [0, 0] and the input point.
#     """
#     x, y = point
#     distance = math.sqrt(x**2+y**2)
#     return distance


# def distance_between_points(point1, point2):
#     """
#     Calculate the distance between two points in a 2D plane.

#     Args:
#         point1 (list or tuple): Coordinates of the first point [x1, y1].
#         point2 (list or tuple): Coordinates of the second point [x2, y2].

#     Returns:
#         float: Distance between the two points.
#     """
#     x1, y1 = point1
#     x2, y2 = point2
#     distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     return distance


if __name__ == "__main__":
    main()
