"""
This script outline the pipeline from camera data to a 3D model to a trajectory for the Widow X arm to follow
"""

"""
1. Take birdseye picture with camera
2. Identify object with keypoint detection
3. Object localization
4. Sweep arm around object for rough view -> use info to create an octree
5. Use octree and optimization algorithm to determine trajectory for follow-up pass
6. Follow path with Widow X until algorithm is sufficently confident in 3D model
"""

