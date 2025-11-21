"""
Taken from opencv.org documentation and modified:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

When launching with python3 you must specify launch parameters
  arg1: Boolean flag for visualization, True means visualise
  arg2: string path to folder containing calibration images relative to file.
        Can be relative or absolute.
"""

import sys

import numpy as np
import cv2 as cv
import glob

VIS_FLAG = sys.argv[1]
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(sys.argv[2])

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print("IMAGE COULD NOT BE READ")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10, 7), None)
    print(f"ret is {ret}")

    # If found, add object points, image points (after refining them)
    if ret == True:
      objpoints.append(objp)

      corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)

      if VIS_FLAG:
        # Draw and display the corners
        cv.drawChessboardCorners(img, (10, 7), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"Camera re-projection error:\n{ret}")
print(f"Camera matrix:\n{mtx}")
print(f"Camera distortion parameters:\n{dist}")


cv.destroyAllWindows()
