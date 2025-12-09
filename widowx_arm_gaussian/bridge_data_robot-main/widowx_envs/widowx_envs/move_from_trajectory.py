#!/usr/bin/env python3

import argparse
import numpy as np
import math
import time
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

class arm_controller():
    def __init__(self):

        self.client = WidowXClient(host='localhost', port=5556)

        self.scanning = True
        self.has_objective = False
        self.reached_objective = False

        self.trajectory = []

        self.initialize_arm()
        self.initial_scan()

        while self.scanning:
            if self.has_objective:
                self.traj_move()
                time.sleep(1)
            else:
                time.sleep(0.1)

    def initial_scan(self):
        # self.client.move()
        # time.sleep(5)
        pass

    def circle_path(self, center_pos):
        # make circle path around x,y,z position
        pass
    
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
        self.client.move(np.array(self.trajectory.pop[0]))
        if len(self.trajectory) == 0:
            self.has_objective = False



def move_from_trajectory(trajectory, client, slpTime):
    # Go to first point
    client.move(np.array([trajectory[0][0], trajectory[1][0], trajectory[2]
                [0], trajectory[3][0], trajectory[4][0], trajectory[5][0]]))
    time.sleep(5)

    # Follow Trajectory
    for point in trajectory:
        client.move(
            np.array([point[0], point[1], point[2], point[3], point[4], point[5]]))
        time.sleep(slpTime)

    print("finished")
    time.sleep(5)

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