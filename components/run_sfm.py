import cv2 as cv
import numpy as np
from sfm import SFM
import yaml


def main():
    # Define two images where motion is happening
    im1 = cv.imread("../olin_labset/sfm_frames/0034.jpg", cv.IMREAD_GRAYSCALE)
    im2 = cv.imread("../olin_labset/sfm_frames/0035.jpg", cv.IMREAD_GRAYSCALE)

    # read intrinsics from yaml and load to camera matrix
    fx = None
    fy = None
    cx = None
    cx = None
    with open("../olin_labset/camera.yaml", "r") as file:
        p_cam = yaml.safe_load(file)
        intrinsics = p_cam["intrinsics"]
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # construct SFM object
    sfm = SFM()

    # 1. run matching
    matches, keys1, desc1, keys2, desc2 = sfm.match_features(im1, im2)

    if matches is None:
        raise Exception("NO MATCHES FOUND")

    im3 = cv.drawMatches(
        im1,
        keys1,
        im2,
        keys2,
        matches[:40],  # choose the number of matches to disp
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # sort points for those that helped with  matches matching
    pts1 = np.float32([keys1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keys2[m.trainIdx].pt for m in matches])

    # check RANSAC train condition, only check pt1 cause len should be same
    if pts1.shape[0] < 16:
        raise Exception("NOT ENOUGH MATCHES")

    # 2. find essential matrix
    R, t, pts1_inliers, pts2_inliers = sfm.get_frame_transformation(pts1, pts2, K)

    print(f"Rotation:\n{R}")
    print(f"transformation:\n{t}")

    cv.imshow("im3", im3)
    cv.waitKey(0)

    # cleanup
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
