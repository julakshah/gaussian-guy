import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def create_point_cloud_from_rgbd(color_path, depth_path, fx, fy, cx, cy):
    # 1. Read color and depth images
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)

    # 2. Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000.0,     # depth in millimeters
        depth_trunc=3.0,        # meters
        convert_rgb_to_intensity=False
    )

    # 3. Visualize RGB + Depth
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("RGB image")
    plt.imshow(np.asarray(rgbd_image.color))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Depth image")
    plt.imshow(np.asarray(rgbd_image.depth), cmap="gray")
    plt.axis("off")

    plt.show()

    # 4. Camera intrinsics
    color_np = np.asarray(rgbd_image.color)
    height, width = color_np.shape[:2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )


    # 5. Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    # Flip for correct visualization
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # 6. Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )

    # Optional: downsample (faster visualization)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Debug info
    print("Point cloud has", np.asarray(pcd.points).shape[0], "points")
    print("Has colors:", pcd.has_colors())

    # 7. Visualize point cloud
    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name="RGB-D Point Cloud",
        width=1024,
        height=768,
        left=50,
        top=50
    )

    return pcd


# --- Example Usage ---

color_image_path = "/home/connor/Downloads/rgbd-scenes/background/background_1/background_1_1.png"
depth_image_path = "/home/connor/Downloads/rgbd-scenes/background/background_1/background_1_1_depth.png"

fx_val = 525.0
fy_val = 525.0
cx_val = 319.5
cy_val = 239.5

point_cloud = create_point_cloud_from_rgbd(
    color_image_path,
    depth_image_path,
    fx_val, fy_val, cx_val, cy_val
)
