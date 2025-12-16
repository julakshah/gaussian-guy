import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'widowx_arm_gaussian/bridge_data_robot-main/widowx_envs/widowx_envs'))

import numpy as np
import cv2
import threading
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class camera_controller():
    def __init__(self):
        print("camera Init")
        self.bridge = CvBridge()

        self.latest_depth = None
        self.depth_lock = threading.Lock()

        self.latest_rgb = None
        self.rgb_lock = threading.Lock()

        if not rospy.core.is_initialized():
            rospy.init_node('camera_controller', anonymous=True)

        rospy.Subscriber('/D435/depth/image_rect_raw',
                         Image, self._depth_callback)
        rospy.Subscriber('/D435/color/image_raw',
                         Image, self._rgb_callback)

    def _depth_callback(self, msg):
        """Callback that receives depth frames from ROS"""
        depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")
        with self.depth_lock:
            self.latest_depth = depth_image

    def _rgb_callback(self, msg):
        """Callback that receives RGB frames from ROS"""
        rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.rgb_lock:
            self.latest_rgb = rgb_image

    def get_depth(self):
        """Returns the latest depth frame as numpy array (uint16, millimeters)"""
        with self.depth_lock:
            return self.latest_depth.copy() if self.latest_depth is not None else None

    def get_rgb(self):
        """Returns the latest RGB frame as numpy array (uint8, BGR)"""
        with self.rgb_lock:
            return self.latest_rgb.copy() if self.latest_rgb is not None else None

    def loop_wrapper(self):
        """Main loop - keep ROS spinning to receive depth and RGB frames"""
        print("Starting camera loop...")
        while not rospy.is_shutdown():
            # depth = self.get_depth()
            # rgb = self.get_rgb()

            # if depth is not None:
            #     depth_colormap = cv2.applyColorMap(
            #         cv2.convertScaleAbs(depth, alpha=0.03),
            #         cv2.COLORMAP_JET
            #     )
            #     cv2.imshow("Depth View", depth_colormap)

            # if rgb is not None:
            #     cv2.imshow("RGB View", rgb)

            #     if cv2.waitKey(1) & 0xFF == 27:
            #         break

            rospy.sleep(0.01)

        print("Closing camera...")
        cv2.destroyAllWindows()