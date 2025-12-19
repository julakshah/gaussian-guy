#!/usr/bin/env python3

import importlib
import time
import math
import numpy as np
import argparse
import threading
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


class arm_controller():
    def __init__(self):
        print("arm Init")

        self.client = WidowXClient(host='localhost', port=5556)

        self.scanning = True
        self.scan_position_reached = False
        self.has_objective = False
        self.reached_objective = False

        self.destinations_reached = 0
        self.last_objective = [0, 0, 0, 0, 0, 0]

        self.trajectory = []

        # Initialize arm
        self.initialize_arm()
        time.sleep(5)

        # Go to initial scan position
        # self.initial_scan()
        # time.sleep(5)

        self.scan_position_reached = True

    def reset_arm(self):
        # Reset after scan position
        self.client.reset()

    def initial_scan(self):
        self.client.move(
            np.array([0.2, 0, 0.3, 0, 1.5, 0]), blocking=True, duration=1)
        self.client.move(
            np.array([0.15, 0, 0.4, 0, 1.5, 0]), blocking=True, duration=1)
        self.client.move(
            np.array([0.25, 0, 0.4, 0, 0.5, 0]), blocking=True, duration=1)
        self.client.move(
            np.array([0.10, 0, 0.525, 0, 5.8, 0]), blocking=True, duration=1)
        self.scan_position_reached = True
        print("Scan Position Reached (controller)")

    def circle_path(self, center_pos):

        points = 36

        if center_pos[0] + 0.15 <= 0.43:
            min_x = 0.125

            x_radius = min(0.15, center_pos[0]-min_x)
            y_radius = x_radius*1.3
            x_height = 0.110
            y_height = 0.015
            yaw = 0.9

            angles = np.concatenate(
                [np.linspace(0, 2*np.pi, points)[int(points/2):points], np.linspace(0, 2*np.pi, points)[0:int(points/2)]])

            xs = np.array(x_radius*np.cos(angles)) + center_pos[0]
            ys = -np.array(y_radius*np.sin(angles)) + center_pos[1]
            rolls = np.array(angles)
            heights = np.array(x_height*abs(np.cos(angles))) + y_height
            pitches = -np.array(yaw*(np.cos(angles))) + \
                1.5  # * abs(np.cos(angles))

            droop_assist = np.array(
                [find_pickup_droop([xs[i], ys[i]]) for i in range(points)])

            for i in range(points):
                self.trajectory.append(np.array(
                    [xs[i], ys[i], heights[i]-droop_assist[i], rolls[i], pitches[i], 0]))

            self.has_objective = True
            print("Circle Path Added")
        else:
            print("Object to far, initial circle skipped")

    def loop_wrapper(self):
        while (self.scanning):
            if (self.has_objective and not self.reached_objective):
                self.trajectory_move()
            else:
                time.sleep(0.1)

    def initialize_arm(self):
        self.client.init(WidowXConfigs.DefaultEnvParams, image_size=256)
        print('Waiting 5s to ensure server fully initialized...')
        time.sleep(5)
        print("Starting robot.")

    def add_to_trajectory(self, point):
        self.trajectory.append(point)
        self.has_objective = True

    def shutdown(self):
        self.scanning = False

    def trajectory_move(self):
        self.last_objective = self.trajectory.pop(0)
        self.client.move(np.array(self.last_objective),
                         blocking=True, duration=0.75)
        # time.sleep(1.5)
        self.reached_objective = True
        if len(self.trajectory) == 0:
            self.has_objective = False

    def next(self):
        self.reached_objective = False


def find_pickup_droop(pickup_point):
    a = distance_from_zero_zero(pickup_point)
    droop = -2.35E-03 + 0.0522*a + -0.216*a**2
    return droop


# def find_placing_droop(pickup_point, place_point):
#     pickup_disp = distance_from_zero_zero(pickup_point)
#     place_disp = distance_from_zero_zero(place_point)
#     xy_increase = place_disp - pickup_disp
#     return (pickup_disp - max(place_disp, 0.2)) * -0.0529


def distance_from_zero_zero(point):
    """
    Calculate the distance between [0, 0] and another point [x, y2.

    Args:
        point (list or tuple): Coordinates of the point [x, y].

    Returns:
        float: Distance between [0, 0] and the input point.
    """
    x, y = point
    distance = math.sqrt(x**2+y**2)
    return distance


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
