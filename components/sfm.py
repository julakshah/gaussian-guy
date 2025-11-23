import numpy as np
import cv2 as cv


class SFM:
    def __init__(self):
        # init ORB algorithm
        self.orb = cv.ORB_create(nfeatures=5000)
        # init Brute Force matcher for
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def match_features(self, im1, im2):
        """
        Runs feature matching on two inputted numpy images.
        Args:
          im1: a opencv image frame IN GRAYSCALE
          im2: a opencv image frame IN GRAYSCALE
        Return
          None (5x): if matching failed, returns None
          matches: A list of DMatch objects returned from BFMatcher.match
          keys1: A list of keypoints from im1
          desc1: A list of descriptors for each keypoints in keys1
          keys2: A list of keypoints from im2
          desc2: A list of descriptors for each keypoints in keys2
        """

        keys1, desc1 = self.orb.detectAndCompute(im1, None)
        keys2, desc2 = self.orb.detectAndCompute(im2, None)

        if desc1 is None or desc2 is None or len(keys1) == 0 or len(keys2) == 0:
            # no features
            # return a display image (draw nothing) and None pts
            return None, None, None, None, None

        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches, keys1, desc1, keys2, desc2

    def get_frame_transformation(self, pts1, pts2, K):
        """
        Finds the transformation matrix between a set of matched points
        Args:
          pts1: A set of keypoints matched with pts2
          pts2: A set of keypoints matched with pts1
          K: The camera matrix of the given frames
        Return
          R: rotation matrix from pts1 -> pts2
          t: transformation vector from pts1 -> pts2
          pts1_inliers: points that were valuable to estimation, corresponding
                        to pts2
          pts2_inliers: points that were valuable to estimation, corresponding
                        to pts1
        """
        # find fundamental matrix (in pixels)
        F, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 3.0)

        # exit if fund matrix couldn't be found
        if F is None or inliers is None:
            print("FUNDAMENTAL MATRIX COULD NOT BE FOUND")
            return np.eye(3), np.zeros((3, 1)), None, None

        # compute essential
        E = K.T @ F @ K

        # build boolean mask of inliers
        mask_bool = inliers.ravel() == 1
        pts1_inliers = pts1[mask_bool]
        pts2_inliers = pts2[mask_bool]

        if pts1_inliers.shape[0] < 5:
            # not enough inliers
            return np.eye(3), np.zeros((3, 1)), None, None

        # recoverPose returns inliers too; we pass K and pixel points
        _, R, t, pose_mask = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)

        #TODO: CHECK THIS
        # normalize translation (direction only)
        t = t / (np.linalg.norm(t) + 1e-9)

        return R, t, pts1_inliers, pts2_inliers

    def triangulate(self, pose1, pose2, pts1, pts2):
    """
    Triangulates pts1 and pts2 in relation to their poses given that poses are
    in world frame.
    Args:
      pose1: 4x4 camera pose matrix of pose1 in world frame
      pose2: 4x4 camera pose matrix of pose2 in world frame
      pts1: a list of 2d points that has index matched correspondences in pts2
      pts2: a list of 2d points that has index matched correspondences in pts1
    Return:
      good_points: corresponded points mapped to 3d space
    """
        if pts1 is None or pts2 is None or pts1.shape[0] == 0:
            return

        # Prepare homogeneous output
        pts3d = np.zeros((pts1.shape[0], 4))

        # Projection matrices: P = K * [R | t]
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]

        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            u1, v1 = p1
            u2, v2 = p2

            # magic linalg that simplifies PX[2] into the equation
            A = np.zeros((4, 4))
            A[0] = u1 * P1[2] - P1[0]
            A[1] = v1 * P1[2] - P1[1]
            A[2] = u2 * P2[2] - P2[0]
            A[3] = v2 * P2[2] - P2[1]

            _, _, vt = np.linalg.svd(A)
            X_hom = vt[-1]  # last row is smallest singular vector
            pts3d[i] = X_hom / (X_hom[3] + 1e-12)  # normalize w to avoid huge numbers

        # Convert to Cartesian
        pts3d_cartesian = pts3d[:, :3]  # already normalized above

        # compute depth in camera frames: z = R * X + t
        z1 = (pose1[:3, :3] @ pts3d_cartesian.T + pose1[:3, 3:4])[2, :]
        z2 = (pose2[:3, :3] @ pts3d_cartesian.T + pose2[:3, 3:4])[2, :]


        good_points = pts3d[mask]

        return good_points

    def fast_pose_inverse(self, pose):
        """
        uses orthogonal property of rotation matrices to speed up inversion.
        Args:
          pose: a 3x4 matrix [R|t]
        Returns:
          pose_inv: a 3x4 matrix inversion of the input pose
        """

        R = pose[:3, :3]
        t = pose[:3, 3]

        pose_inv = np.zeros((3, 4))
        pose_inv[:, :3] = R.T
        pose_inv[:, 3] = -R.T @ t

        return pose_inv

    def add_ones(self, pts):
        """
        Helper function to add a column of ones to a 2D array (homogeneous
        coordinates).
        """
        return np.hstack([pts, np.ones((pts.shape[0], 1))])
