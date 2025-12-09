#!/usr/bin/env python3

import argparse
import numpy as np
import math
import time
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from move_from_trajectory import move_from_trajectory


def main():
    captures = 0
    parser = argparse.ArgumentParser(
        description='WidowX robotic arm for 3D imaging with gaussian splatting')

    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)

    parser.add_argument('--repeats', type=int, default=4,
                        help='Number of back-and-forth repeats')
    parser.add_argument('--sleep', type=float, default=1,
                        help='Time to sleep between moves (seconds)')
    parser.add_argument('--pose1', type=float, nargs=6,
                        default=[0.1, 0, 0.15, 0, 1.5, 0], help='First pose')
    parser.add_argument('--pose2', type=float, nargs=6,
                        default=[0.1, 0.05, 0.15, 0, 1.5, 0], help='Second pose')

    args = parser.parse_args()

    # Initialization
    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)
    print('Waiting 5s to ensure server fully initialized...')
    time.sleep(5)
    print("Starting robot.")

    # Initial Reset
    print('Resetting robot to neutral')
    r = client.reset()
    print('Reset status:', r)
    time.sleep(2)

    radius = .1
    angles = np.linspace(0, 2*np.pi, 20)

    xs = np.array(radius*np.cos(angles)) + 0.3
    ys = -np.array(radius*np.sin(angles))
    rolls = np.array(angles)

    trajectory = np.empty((23,), dtype=object)

    for i in range(10):
        trajectory[i] = np.array(
            [xs[i], ys[i], 0.025, rolls[i], 1.5, 0])
    for i in range(3):
        trajectory[i+10] = np.array(
            [xs[9], ys[9], 0.025, (rolls[9]-((i+1)*rolls[9]/3)), 1.5, 0])
    for i in range(10):
        trajectory[i+13] = np.array(
            [xs[i+10], ys[i+10], 0.025, rolls[i+10], 1.5, 0])
        
    time.sleep(5)
    move_from_trajectory(trajectory,client,0.25)

    print("finished")
    time.sleep(5)

    # Final Reset
    print('Resetting robot to neutral')
    r = client.reset()
    print('Reset status:', r)
    time.sleep(2)

# def get_tf_mat(pose):
#     # convert pose to a 4x4 tf matrix, rpy to quat
#     quat = quaternion_from_euler(pose[3], pose[4], pose[5])
#     tf_mat = quaternion_matrix(quat)
#     tf_mat[:3, 3] = pose[:3]
#     return tf_mat


def find_pickup_droop(pickup_point):
    a = distance_from_zero_zero(pickup_point)
    droop = -2.35E-03 + 0.0522*a + -0.216*a**2
    return droop


def find_placing_droop(pickup_point, place_point):
    pickup_disp = distance_from_zero_zero(pickup_point)
    place_disp = distance_from_zero_zero(place_point)
    xy_increase = place_disp - pickup_disp
    return (pickup_disp - max(place_disp, 0.2)) * -0.0529


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


def distance_between_points(point1, point2):
    """
    Calculate the distance between two points in a 2D plane.

    Args:
        point1 (list or tuple): Coordinates of the first point [x1, y1].
        point2 (list or tuple): Coordinates of the second point [x2, y2].

    Returns:
        float: Distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


if __name__ == "__main__":
    main()
