# Gaussian Guy

## Introduction

This project was created for the Computational Robotics Fall 2025 Final.

We aimed to create a Gaussian Splatting Reconstruction of an object using image data taken from a robot arm navigating around it. Our project MVP included:

- Generating waypoints for the robot end effector to travel to.
- Taking a set of images of a given object at various angles to generate training and testing data for our splatting implementation.
- Running Gaussian Splatting on this dataset using gsplat to create a 3D mesh of the object.

Our final product is divided into two stages. Our first stage will be the robot arm taking in x, y, z coordinates and roll, pitch, yaw values from our waypoint generation algorithm and taking images of the object from those angles. Our second stage will be inputting these training images into gsplat and and generating a mesh reconstruction of our test object.

To view more about the development process and milestones about our project, view our website [here](https://julianshah.com/gaussian-guy/)

## Project Structure

This project leverages two major open-source components: documentation from a previous [CompRobo Final](https://github.com/eddydpan/chess_bot) which referenced the Berkeley Robot and AI Lab and [gsplat](https://github.com/nerfstudio-project/gsplat), a library for efficient Gaussian Splatting.

Along with this, we implemented a method of generating the best path around a given object to add the most information to our data set using an A* algorithm, a method to cluster our splats, and a method to mask out and change the opacity of the splats  and isolate an object from the background.

### Directory Structure



### Setup



### Execution

