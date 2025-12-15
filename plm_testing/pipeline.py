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

"""
Adaptive planning loop:
1. Take and save image with camera (RGBD) -> used for gaussian later
2. open3d.geometry.create_point_cloud_from_rgbd_image() to generate point cloud
3. Use insert_obstacle to update octree -> may be better with Open3D octree instead?
4. Use raycasting to identify empty cells between robot and obstacle
5. Calculate distance to nearest unknown cell
6. Use A* to find path to the nearest unknown cell (with a standoff to avoid collision)
7. Send path as waypoints for Widow X to follow
8. Update octree to fill in occupied cells if surrounded by occupied cells
9. Repeat 1 through 8 until all unknown cells are cleared
"""

import open3d

from plm_testing.A_Star_octree import go_to_goal, find_unknown_leaves
from plm_testing.Octree import OctreeNode

octree = OctreeNode(position=(0,0,0), r=16, min_r=0.5)
unknowns_remain = True
while unknowns_remain:
    # -> call camera API to get image
    point_cloud = open3d.geometry.create_point_cloud_from_rgbd_image(image, intrinsics, extrinsic) # does extrinsics send it to world frame?
    camera_pose = # -> get_camera_pose()
    point_cloud.transform(camera_pose)
    down_point_cloud = point_cloud.voxel_down_sample(voxel_size=2.0)
    start_pos = # -> get_robot_position() -> controller.last_objective <- is the actual call

    for point in down_point_cloud.points:
        octree.insert_obstacle(tuple(point))
        ## raycast to update with empties
        octree.raycast(start_pos, tuple(point))

    path = go_to_goal(octree, start_pos)
    for waypoint in path:
        # -> move_to(waypoint)
        pass
    unknown_leaves = []
    find_unknown_leaves(octree, unknown_leaves)
    if not unknown_leaves:
        unknowns_remain = False

print("Object scanned")