# Gaussian Guy

### Ananth, Ben, Connor, Grant, Julian


## Project Overview (main idea, motivation, mvp, stretch goals)

The goal of this project is to visually reconstruct a 3D object via Gaussian splatting, using data recorded by a robotic arm equipped with a monocular camera. We imagine the robotic arm taking a steady 360-degree video around an object before feeding that into a Gaussian splatting algorithm to compute the Gaussian ellipsoids before displaying the processed image (we’re thinking Blender or Pangolin because they natively support 3D rendering of point clouds without too much work.) The output of our pipeline will be a 3D representation of a scene measured by a camera attached to a robotic arm, which we can then feed into any other metrics or algorithms we deem fitting.

Our team chose this project because we had a series of different interests and this project blends those together. Some people wanted to work with the robotic arm, some people wanted to work on more computer vision or 3D reconstruction algorithms, and some of us wanted to do a project that physically interacted with the world, as our camera does when measuring the object. We’re also interested in how we can play around with the results of Gaussian splatting, given that it outputs a physically-meaningful representation of the scene, which means that doing some processing on the resulting scene doesn’t seem impossibly hard. This project blends those aspects together and gives everyone on the team a part to play. 

Some potential applications of this project include high-quality reconstruction of a 3D object, photogrammetry, and 3D asset generation. We could envision other tangential applications as well, even if this project wouldn’t be the best suited for them — for instance, we could compare two ideally identical objects by taking the magnitude of the differences of the Gaussians describing the scene after doing some mean-centering and alignment of the objects. Additionally, as a GS representation of a 3D scene consists of a large number of visualization primitives in space, any operations that can be performed on a point cloud should also be applicable to a GS scene, such as clustering algorithms to segment specific objects. 


## Milestones

**Rendering/reconstruction (Gaussian splatting)**

By the first milestone, from the CV/reconstruction side of the project, we want to have COLMAP or some similar structure-from-motion pipeline working on our own video source, perhaps just on a phone or laptop camera moving in space. We’ll also have spent time reading about Gaussian splatting and looking into the papers describing its implementation so that we have a clear idea about how to proceed with it.

By the second milestone, we aim to have the existing Gaussian splatting implementation that we’re aware of working on a test dataset, and already have progress processing our own video into a format compatible with the Gaussian splatting algorithm. This will set us on track for connecting that video preprocessing step to the video from the camera on the arm, and then piping that into the Gaussian splatting algorithm. We’ll also have figured out how to visualize the result of Gaussian splatting. If any of the stretch goals (i.e. performance metrics versus ground truth) seem within reach, we’ll have a plan for these as well.

**Robot control (kinematics and trajectory generation)**

By our first milestone, we will finalize how we are tackling this project and start exploring both the ROS framework needed as well as basic trajectory generation. As ROS has MoveIt we can focus on the trajectory aspect, but the actual control of the arm is unknown and will be good to start looking at as early as possible. We will also start exploring how we can implement some of our stretch goals, such as object avoidance and path optimization.

By our second milestone, we plan to have the basic functionality of the robot arm working. By this point, we will be content with the ability to complete a predetermined circle around our object. Once this is complete, we can start experimenting with path optimization for our arm’s movement for different objects.


## MVP/Stretch

The MVP is getting a full pipeline working using existing implementations and frameworks. This would be like using Move-it for robot movement, a split trajectory generation library, a Gaussian splatting implementation like [gsplat](https://docs.gsplat.studio/main/) and then writing some code to pipe that into a visualization software, potentially with some performance metrics using static video frames as ground truth. 

The stretch goals depend on the component of the project:

**Robot control (kinematics and trajectory generation)**

- Detecting an object and creating a path around the object in real time (some form of machine vision)

- Creating our own end-effector-centric trajectory algorithm

- Creating our own numerical inverse kinematics script

- Creating our own analytical inverse kinematics script

**Scene rendering (Gaussian splatting and parallel process)**

- Generating metrics by comparing ground-truth (static video frame) to reconstruction (Gaussian splats from a viewpoint the same as the video frame.)

- Scene editing? (described a little [here](https://arxiv.org/pdf/2404.13679) and [here](https://arxiv.org/pdf/2410.12262v2)). If we can use clustering algorithms like DBSCAN to identify an object from our list of Gaussians, we can try and see the visualization of isolating removing that object from the scene. Or, if we’re curious, we could take all the Gaussians in that cluster and do something like halve their opacity. 

- Comparison of objects/scenes by aligning point cloud points, taking difference between relevant Gaussians, etc. (e.g. cluster point cloud points, zero the means of the Gaussians by the dominant cluster means, compute moments of the cluster opacity to align the clusters, and compute a descriptor somehow for the part of the scene around the object in order to compare objects between scenes?) 

- Writing any portion of the Gaussian splatting algorithm ourselves (SFM, ellipsoid init, etc. [REF](https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/)). This could also include writing some code to run in parallel (CUDA or non-CUDA shaders to run on a GPU, for instance.)


## Learning Goals

Ananth: Through this project, I will learn the fundamentals of robot control, including concepts like inverse kinematic math and trajectory generation. My goals include understanding the basics and expanding into more complex algorithms, so I will follow a similar path through this project.

Ben: Through this project, I want to become familiar with the math behind Gaussian splatting and structure-from-motion, as well as understand the primary components of Gaussian splatting that are the most expensive to implement. In particular, I want to explore the optimization that GS implementations use, and how the parameters of Gaussians to add after each optimization are determined from the cost/reward function they use. I’d also like to explore the potential applications/uses of a scene created by Gaussian splatting — thus, the stretch goals about generating metrics or doing some sort of clustering on the result. Ideally, aligned with my goals for the course, this project will give me a solid overview of an algorithm not specifically explored in class (one of my goals), teach me about potential ways to take it further in the future (another goal — get ideas for other things to explore in the future), and give me an opportunity to write neat or organized code, as has been a consistent goal of mine throughout the class.

Connor: This project aligns my interests in using C++, learning more about controls, and will allow me to create a diagram of the perception system.

Grant: There are a few things I want to get out of this project, I want to get the ROS2 experience using a different robot (arm vs Neato), I want to gain an understanding of the trajectory generation process, and I want to gain a more basic understanding of the analytical inverse kinematics process. As my learning goals are sort of based on this project, it of course aligns well with them, but overall, almost every part of this project is incredibly interesting to me.

Julian: I want to gain more experience with the algorithmic side of robotics (specifically, for this project, the Gaussian splatting component).


## Algorithms to explore & Develop (topics to explore)

Algorithms: Structure from Motion, Gaussian Splatting, Analytical Inverse Kinematics, Trajectory Generation

To demonstrate these algorithms, we’ll generate a 3D reconstruction of an object using Gaussian splatting, with the video taken from a robotic arm following some generated trajectory. Our deliverables will explain the high-level aspects of the trajectory generation and 3D reconstruction, specifically mention the work we’ll do in stitching these elements together, and will present some way of measuring the performance of our reconstruction against the ground truth, be that a side-by-side visual comparison or an actual metric.


### Gaussian Splatting Resources

- [Gsplat implementation](https://docs.gsplat.studio/main/)

- [A walkthrough blog on Gaussian splatting concepts](https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/)

- [Original Paper on Computationally Efficient Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

- [Feature Consistent Gaussian Splatting for Object Removal](https://arxiv.org/pdf/2404.13679)

- Other potential terms/topics of interest:

  - COLMAP (SfM)

  - Clustering algorithms (for example, DBSCAN)

  - Alignment of point clouds (PCA, moments of inertia)

  - Mapping between scale + rotation of ellipsoid to covariance matrix of Gaussian

  - Distance metrics for point clouds (for instance, [Wasserstein metric](https://arxiv.org/pdf/2102.04014))


### Robotic Arm Control / Kinematics Resources

- [Fund Robo Curriculum](https://funrobo-olin.notion.site/)

- [Advanced Trajectory Generation](https://www.nature.com/articles/s41598-025-14801-7)

- [Robotic Arm Toolbox](https://petercorke.github.io/robotics-toolbox-python/arm_trajectory.html)


## Challenges (risks)

- Getting the arm to move at all (unknown software using ROS2)

- Generating a path in real time to adjust for object inconsistencies

- Getting Gaussian splatting to render (hard to test since each attempt can take 1h+)

  - One of the larger risks — implementation is a bit of a black box despite the high-level approach making sense, so debugging it may be difficult if it becomes necessary

- Making sure the dependencies for all of our packages sync up properly and we don’t have version conflicts

  - Getting it to run on each of our computers ideally

- Performance metrics may be difficult to interpret given high-dimensional data (small misalignment could give huge error)

- **Debugging…** 


## Teaching Team Asks

- Might need help with the ROS control of the robotic arm

- Any tips on preprocessing for SfM or Gaussian splatting, or any other potential hurdles involved?
