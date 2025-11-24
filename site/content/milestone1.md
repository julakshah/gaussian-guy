# Milestone 1

## MVP and Stretch Goals

Our minimum viable product will consist of:

- Interfacing with the robot arm to generate and execute a trajectory
- Automatically piping video frome a camera on the arm's end effector to a script that will isolate the frames and pass them into a structure-from-motion algorithm and a high quality 3D reconstruction and rendering algorithm. Currently, our plans are to use either COLMAP or our own implementation of SfM for generating a point cloud, and using either Gaussian Splatting with [gsplat](https://github.com/nerfstudio-project/gsplat) or [Triangle Splatting](https://trianglesplatting.github.io/) for the full reconstruction.
- Post-reconstruction scripts to measure the visual error between our 3D reconstruction and ground truth at different angles, as well as perform a non-trivial operation on the geometry primitives of the reconstruction to demonstrate our own ability to work with the results of the reconstruction. Current ideas for this are either isolating a specific object in the reconstruction, applying some visual effect to a specific object in the reconstruction, or trying to match objects. 

**Specific Changes**: Primary changes from our proposal are the consideration of Triangle Splatting as a potential alternative to Gaussian Splatting, as well as the movement of processing on our 3D reconstruction from being entirely stretch goals to our MVP (specifically, that we do at least one sort of processing on the result) as this aligned more with our own interests and learning goals which would otherwise not have been served as well by solely stitching together existing code. We also added the consideration of sparse reconstruction data in our trajectory generation algorithms as a stretch goal, given it better encourages us to integrate our components together.

Stretch goals for this include:

- Creating a path around an object in real time for the arm
- Trajectory generation / optimization that takes input from the sparse reconstruction real-time to better observe the object (NEW --- added as a stretch goal because it puts greater emphasis on integration of our different components)
- Creating our own scripts for solving inverse kinematics of the arm numerically or analytically
- Scene editing per [this source](https://arxiv.org/pdf/2404.13679) and [this source](https://arxiv.org/pdf/2410.12262v2), where we isolate an object and remove it from the reconstructed scene.
- Comparison of objects/scenes by aligning point cloud points, taking difference between relevant Gaussians, etc. (e.g. cluster point cloud points, zero the means of the Gaussians by the dominant cluster means, compute moments of the cluster opacity to align the clusters, and compute a descriptor somehow for the part of the scene around the object in order to compare objects between scenes?) 
- Writing any portion of the Gaussian splatting algorithm ourselves (SFM, ellipsoid init, etc. [REF](https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/)). This could also include writing some code to run in parallel (CUDA or non-CUDA shaders to run on a GPU, for instance.)

## Status

We've successfully interfaced with the robot arm we're planning to use, controlling it via a script. 

On the reconstruction side, we have already tested COLMAP on a dataset of images to examine its reconstruction, and are currently working on the environment setup for gsplat and triangle_splatting in parallel to determine which we will go forward with using. We're running into a few issues installing the necessary versions of packages, especially when CUDA is involved, but are working through those. 

## Project Components

Regarding our hardware, we have the robotic arm successfully moving and have connected to it via a computer.

On the computational side, we have the beginning of our pipeline --- a quick script to process a video and write out every frame to a directory of sequential images, which can then be used as a target for SfM and the rest of our project. 

As we're considering using a triangulated point cloud generated via visual odometry in informing our trajectory decisions when scanning the object, we have an initial draft of a simple visual odometry script implemented, using OpenCV's ORB keypoint detection and matching. 

In parallel to this, we've been ensuring that we can stitch together existing algorithms for this pipeline to evnetually do our own processing at the end. For this purpose, we have COLMAP working and generating a sparse reconstruction, dense reconstruction, and Delaunay-triangulated mesh from a sequence of images. 

Here's a dense reconstruction of a test dataset using COLMAP:
![Dense Reconstruction via COLMAP](/images/colmap-test-dense.png)

We also were able to use COLMAP in generating a Delaunay-triangulated mesh from this reconstruction:
![Mesh via COLMAP](/images/colmap-test-mesh.png)

While this is visually nice to see, our end goal for the project is to have a visual reconstruction based off of Gaussian splatting or triangle splatting, which are somewhat harder to set up. However, COLMAP excels at initially generating a sparse reconstruction in relatively quick time, which we can try to use in place of or along side a home-baked SfM approach if it proves more efficient. 

## Potential Risks

Our most significant risk was being unable to connect to and interface with the arm, but we've successfully averted this.

Our next greatest risk is in setting up the Gaussian splatting code, as we've run into a number of dependency and version while setting up the environment. While we're confident we can eventually get it working --- one person on our team already has some experience with Gaussian splatting --- the risk is more in the fact that before we run gsplat, we can't easily develop the processing algorithms that will follow for measuring error or isolating objects. This processing dependent on knowing the shape and format in which the list of Gaussians is stored after splatting concludes, and while we can look at the documentation for examples on what this looks like in code, having the output of gsplat on a test dataset in front of us would make testing our later processing code much easier.

## Goals for Milestone 2

By Milestone 2, we aim to have Gaussian splatting (or Triangle Splatting, if we decide to pivot to that) fully working on our machines, and to be able to run these on any video of our choosing. We'll also have a script that automates this process such that the only necessary input is the path to the chosen video, and we'll have at least a preliminary post-splatting script that describes some features of the output --- for instance, number of Gaussians, or the results of a clustering algorithm like DBSCAN on their positions --- as, at minimum, a stand-in for the sort of processing we intend to do on the output, which would operate on each Gaussian primitive as if it were a point cloud point. This leaves the time between Milestone 2 and the final deliverable date for finishing the final processing algorithms and helping our point cloud or sparse reconstruction interface with the arm and path planning code.

Regarding the robot arm, we intend to be able to describe a specific trajectory --- for instance, a circle around the center of the viewing area --- and have the arm execute it, as well as look into potential avenues for optimizing our trajectory generation against some constraints or to maximize information gained from viewing our object. 