#!/usr/bin/env python3

import threading
import argparse
import numpy as np
import math
import time
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from move_from_trajectory import arm_controller


def main():

    controller = arm_controller()

    while (controller.scan_position_reached != True):
        print("while loop")
        time.sleep(0.5)

    print("Scan Position Reached (controller)")
    controller.circle_path([0.3, 0])

    test_thread = threading.Thread(target=controller.loop_wrapper)
    test_thread.start()

    while (controller.has_objective):
        if (controller.reached_objective):
            controller.destinations_reached += 1
            print(f"Objective {controller.destinations_reached} reached")
            time.sleep(2)
            controller.reached_objective = False

    controller.shutdown()

    # Initialization
    # client = WidowXClient(host=args.ip, port=args.port)
    # client.init(WidowXConfigs.DefaultEnvParams, image_size=256)
    # print('Waiting 5s to ensure server fully initialized...')
    # time.sleep(5)
    # print("Starting robot.")

    # Camera Test

    # x limit test
    # client.move(np.array([0.15, 0, 0.15, 0, 1.5, 0]))
    # time.sleep(3)
    # client.move(np.array([0.3, 0, 0.15, 0, 1.5, 0]))
    # time.sleep(3)
    # client.move(np.array([0.45, 0, 0.15, 0, 1.5, 0]))
    # time.sleep(3)
    # client.move(np.array([0.6, 0, 0.15, 0, 1.5, 0]))
    # time.sleep(3)

    # print("finished")
    # time.sleep(5)

    # Final Reset
    # print('Resetting robot to neutral')
    # r = client.reset()
    # print('Reset status:', r)
    # time.sleep(2)


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
