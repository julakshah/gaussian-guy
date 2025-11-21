import cv2 as cv
import numpy as np
from sfm import SFM


def main():
    # Define two images where motion is happening
    im1 = cv.imread("../olin_labset/sfm_frames/0034.jpg", cv.IMREAD_GRAYSCALE)
    im2 = cv.imread("../olin_labset/sfm_frames/0035.jpg", cv.IMREAD_GRAYSCALE)

    sfm = SFM()
    matches, keys1, desc1, keys2, desc2 = sfm.match_features(im1, im2)

    if matches is None:
        raise Exception("NO MATCHES FOUND")

    im3 = cv.drawMatches(
        im1,
        keys1,
        im2,
        keys2,
        matches[:20],  # choose the number of matches to disp
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # build arrays only from valid matches
    pts1 = np.float32([keys1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keys2[m.trainIdx].pt for m in matches])

    cv.imshow("im3", im3)
    cv.waitKey(5000)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
