# Gaussian Guy

## Introduction

This project was created for the Computational Robotics Fall 2025 Final.

We aimed to create a Gaussian Splatting Reconstruction of an object using image data taken from a robot arm navigating around it. Our project MVP included:

- Generating waypoints for the robot end effector to travel to.
- Taking a set of images of a given object at various angles to generate training and testing data for our splatting implementation.
- Running Gaussian Splatting on this dataset using gsplat to create a 3D mesh of the object.

Our final product is divided into two stages. Our first stage will be the robot arm taking in x, y, z coordinates and roll, pitch, yaw values from our waypoint generation algorithm and taking images of the object from those angles. Our second stage will be inputting these training images into gsplat and and generating a mesh reconstruction of our test object.

To view more about the development process, milestones, and project structure about our project, view our website [here](https://julianshah.com/gaussian-guy/)

## Project Structure

This project leverages two major open-source components: documentation from a previous [CompRobo Final](https://github.com/eddydpan/chess_bot) which referenced the Berkeley Robot and AI Lab and [gsplat](https://github.com/nerfstudio-project/gsplat), a library for efficient Gaussian Splatting.

Along with this, we implemented a method of generating the best path around a given object to add the most information to our data set using an A* algorithm, a method to cluster our splats, and a method to mask out and change the opacity of the splats  and isolate an object from the background.

### Directory Structure

#### calibration
`calibration` contains the python file for camera calibration, a process that gives you information regarding the camera being used, such as the intrinsics, which are used for processes relating to the camera.

#### gsplat
TBD

#### images (will become two folder I assume TBD)
TBD

#### src
`src` contains all the python files used for the project with a couple exceptions such as the `calib_camera.py` file. Within `src` are the files used for general control (`run_arm.py`, `run_arm_just_circle.py`), our manipulation of the arm (`arm_controller.py`), collection of data from the realsense camera (`camera_controller.py`), adaptive path finding (`octree.py`, `pipeline.py`, `a_start_octree.py`), video processing (`process_video.py`), and gaussian processing (`process_gaussians.py`, `metrics.py`).

#### widowx_arm
`widowx_arm` contains the files needed to communicate with the Trossen WidowX robot arm and realsense camera. These have been adapted from [Chess Bot](https://github.com/eddydpan/chess_bot), the previous CompRobo final mentioned previously, which adapted the files from [Bridge Data Robot](https://github.com/rail-berkeley/bridge_data_robot), a project by PhD students at UC Berkley.

### Setup
GAUSSIAN SETUP STUFF

For the widowx_arm, the setup can be found within the `README.md` file within `widowx_arm/bridge_data_robot-main`.


### Execution
Once setup has been complete, follow the steps below:

STEP 1:
```bash
# run this in bridge_data_robot_main
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

STEP 2:
```bash
# run this in a separate bridge_data_robot_main
docker compose exec -d robonet bash -lic "python3 /home/robonet/widowx_envs/widowx_envs/widowx_env_service.py --server"
```

STEP 3:
```bash
# run this in the same terminal as the last step
docker compose exec robonet bash
```

STEP 4:
```bash
# run this within the docker container created by the previous step
bash -lic "python3 /home/robonet/host_src/src/run_arm_just_circle.py"
```

This should result in the the head of the arm orbiting the the point 27.5 centimeters forward from the center of the arm's base. Once this loop has been complete, the `images` folder will be cleared and the new images will be saved.

EXECUTION OF SPLAT STUFF