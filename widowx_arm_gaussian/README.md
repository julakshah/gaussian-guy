# Chess-Bot

## Welcome!

Welcome to the Chess-bot landing page. This repository hosts a ROS package that enables a WidowX-200 robotic arm to play chess in real-time against a human player. The package allows the robot to:

- Interpret the changing chessboard using machine vision.
- Calculate the optimal move.
- Physically pick up and place chess pieces to execute its move.

---

## Meet the Team

- **Will Young**  
  Will is a mechanical engineering student at Olin College of Engineering (Class of 2026). He is enthusiastic about robotic arm control, path planning, and ensuring a robust architecture for the project. While not a chess master, he claims watching Gotham Chess twice makes him an expert.

- **Mia Chevere**  
  Mia is an electrical and computer engineering student at Olin College of Engineering. She’s excited about the kinematics math involved in moving the arm to precise locations—despite never having taken basic geometry!

- **Eddy Pan**  
  Eddy is an engineering with computing student at Olin College of Engineering. A former chess prodigy (4th grade), Eddy has dreamed of creating a chess bot ever since declaring his major. He’s determined to make a safer, better chess bot than those that break fingers.

- **Dan Park**  
  Dan is an engineering with computing student at Olin College of Engineering (Class of 2025). He reached the round of 16 in the JPMorgan Chess Competition, so he’s “okay” at chess.

- **Kate McCurley**  
  Kate is an engineering with robotics student at Olin College of Engineering (Class of 2025). Kate is most excited about working on computer vision as well as getting to work with a robotic arm for the first time. 

---

## Package Structure

This ROS package is organized into the following nodes:

### **1. Perception**
- Captures camera input to identify which chess pieces occupy which squares on the board.

### **2. Movement Calculator**
- Uses information from the perception node to recreate the board and determine the robot's next move.
- The move is calculated using discrete math inspired by Eddy’s final exam.
- Sends the move decision to the Path Planner node.

### **3. Path Planner**
- Utilizes the known relationship between the chessboard and the Widow X arm to estimate the pose of each square.
- Generates waypoints to guide the arm along its trajectory.

### **4. Arm Movement Controller**
- Guides the Widow X to move along waypoints and adjusts its grip as needed to pick up and place chess pieces.

---

## Milestone 1

After the first week, the team made significant progress on three key fronts:

### **1. Computational Setup**
- Integrated the **robot bridge repository** from the Berkeley Robot and AI Laboratory to control the Widow X.
- Set up a functional simulator using the **Trossen Robotics X Series Arms Simulator** to enable testing without physical access to the robot.
- Created a collection of helpful commands for controlling the arm.
- Established a plan for deliverables by the next milestone.

### **2. Background Research**
- Researched previous uses of the Widow X, prior chess bots, and kinematic arm theory.
- Gained insights into:
  - Structuring ROS packages for the Widow X.
  - Parallel vs. forward kinematics.
  - Machine vision depth solutions.

### **3. Structure Plan**
- Developed a preliminary system architecture, detailed in the "Package Structure" section above.
- Focused on scoping the Minimum Viable Product (MVP) effectively. For example, alternative methods for pose estimation are prioritized over implementing stereo vision depth estimation.

### **4. Next Steps**
We aim to accomplish a few toy problems that will segue nicely into our project. We will focus on a pick-and-place module and a "move to (x,y,z)" module, which once integrated together alongside a basic chess engine, will act as our MVP. 

### **5. Website**
- Successfully launched this website! You’re looking at it now!

While many details of this project are still in development, the team has a solid foundation and is eager to tackle the challenges ahead.
---

## Milestone 2

### Computational Setup

At the end of Milestone 1, we had a computational setup that sometimes worked, but was extremely finicky and unpredictable. During this milestone, we were able to convert this setup into one that can run the same way every time. 

The main problem we were running into was that the repository we were following required a RealSense camera to properly run the server for the robot, but we were having issues properly configuring our camera while running our docker container. In order to solve this, we modified the code that ran the server and removed all camera topics (and any other code snippets that required them), and we are now planning to use a different (not RealSense) camera and read from it separately. This fix allowed us to finally start testing Python executable files on the physical robot. 

#### Pick and Place

The library we used for our computational setup has several built-in functions for moving the robot. There is a move() function, which moves the end effector to a specified pose in Cartesian space (, y, z, roll, pitch, yaw), a step_action() function, which moves the end effector pose incrementally by the amounts specified, and a move_gripper() function that opens and closes the gripper.

Using the functions specified above were able to implement a basic pick and place function on our robot. This function takes a start point, end point, the height required to grasp the object, and the clearance height required for the move, and the robot then picks up an object located at the start point and places it at the end point. The function also contains a parameter for the maximum allowable gripper width for picking and placing things in tight spaces.

A demonstration of this function is below.

https://drive.google.com/file/d/1UlycNBd_-v8WeDgeD59zWNMrYerPvkAA/view?usp=drive_link

#### Four Corners

In addition to Pick-and-Place, we also wrote a script that would move the end effector to each of the four corners of the chess board. Our goal with this script was to make sure that the arm had the capability to move its end effector to all the positions on our physical chess board.

Through initial testing with running files on the physical robot, we’ve been able to learn

1. Interbotix Python API basic commands such as `move()` and `step_action()`
    - We are all new to the Interbotix python libraries. After completing Pick-and-Place and Four-Corners, we now have a better idea of how we’ll configure the robot to move to each square.
2. Compatibility between arm’s gripper and chess pieces
    - We were worried that the arm would struggle to pick up chess pieces due to their cylindrical shape and that the gripper would be too wide to pick up individual pieces. Nevertheless, in the video above, we demonstrated its ability to pick up pawns in between other pieces.
3. Limits of the arm with our physical chess board
    - We were concerned that the physical arm wouldn’t be able to reach the tiles furthest from its base. We found out that when setting move commands, the positions of the furthest tiles would exceed the joint limits. To address this issue, we’ve raised the arm’s apparatus to overlap with the side of the chess board. Now, our end effector should be able to reach all possible positions on the chess board.

### Simulator

For the simulator with arm, we wanted to achieve live simulation both with and without the robot attached to the computer. We worked on the simulator as a fail-safe in case the team could not get the physical arm running and possibly for multiple people to be working on the arm at the same time.

#### With Connected Arm

We were able to run the connected arm with the simulator with the instructions in a separate tab:

[Simulator with Arm](https://www.notion.so/Simulator-with-Arm-153ad90bdc6980a7882ccef9302fe745?pvs=21) 

The commands open up an RViz display with the arm connected and the command moves the arm when arm commands are inputted through the control panel on RViz.

#### Without Connected Arm

We were able to run the connected arm remotely without needing to connect with the arm with the following command:

```bash
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx200 use_sim:=true
```

However, since we have the MVP for the physical WidowX bot working, we have decided to move away from using the simulator and using the physical bot.

### AprilTags

We also began working on detecting april tags and determining their positions using the python apriltag library. We wrote a script able to detect multiple april tags from a camera feed and get their IDs and position in the image. On the left is a screenshot of the output video feed, which outlines and IDs each april tag. On the right is part of the terminal output with the location of one of the tags, including the position of its center and corners. This information is printed about each tag and is constantly updating with their live positions. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/97f58b88-700b-4d20-ae0c-5ff866afa50b/0a3cb867-3747-41cd-a9c3-9f2148bec202/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/97f58b88-700b-4d20-ae0c-5ff866afa50b/60cb99d3-ad1a-4d33-b9e0-9ba36d63d99d/image.png)

### Kinematic Math!

Even though the kinematic math was abstracted by the Inter box API we still developed the flowing blog post that explains how the WidowX works. We felt that it was important to understand what was going on behind the scenes even if were weren’t manually completing the calculations. 

[The WidowX as a kinematic arm ](https://www.notion.so/The-WidowX-as-a-kinematic-arm-157ad90bdc6980918a5ad019abdbac9c?pvs=21)

### Looking forward!

Though we made good progress in the latest sprint we still have a ways to go! In order to make sure we stay on track we decided to break our remaining tasks into MVP task and stretch tasks.

For our MVP we would like to accomplish the following:

- [ ]  Get everyone computationally set up
- [ ]  Camera module
- Figure out where we want to put april tags (on pieces? chessboard corners?)
- [ ]  MVP: 8x8 array of robot arm poses for each square
- [ ]  MVP: pickup heights for each piece
- [ ]  MVP(?): figure out how to pick up pieces on very corners of board without arm collapsing
- [ ]  MVP: GUI to integrate chess engine
    - [ ]  convert engine move to physical move
    - [ ]  figure out which move the opponent made
    

In our MVP our robot would only be using the camera to identify the opponents move. The robot would then calculate it’s move and execute it by going to pre-saved poses. 

For our stretch goals we would like to look at the following:

- [ ]  April tags on edge of board to get a geometric map of the chess board (known distances between each square, pieces can be abstracted to the center point of each square)
- [ ]  array of poses for graveyard
- [ ]  figure out how to read move status from server so we dont’ have to do janky time.sleep stuff
- [ ]  Camera
    - [ ]  select which camera we’re using
    - [ ]  Post-MVP
        - [ ]  mounting location for the camera
        - [ ]  physical camera mount design/fab(?)
        - [ ]  functional AprilTag code in Python: geometric poses between april tags (on 4 corners of chess board) relative to camera
        - [ ]  AprilTag code for transforming camera frame to arm frame
        - [ ]  post post post mvp: finger detection
    

### System Architecture 

<img src="/sys_arch.jpg">


The diagram above shows all the core components that make up the Chess Bot and the data that flows between them. Further descriptions of each component are below.

**base_env:** Initializes a class for a high-level base environment for robotics or simulation applications.

**robot_base_env:** Initializes a class that inherits from base_env with functionality specific to robotics applications.

**controller_base:** Initializes abstract base classes for creating a controller for a robotic arm and a controller for a gripper end effector.

**custom_gripper_controller:** Initializes a class for controlling the default WidowX end effector that inherits from GripperControllerBase.

**widowx_controller:** Initializes a class for controlling the full WidowX robot arm. This class inherits from the RobotControllerBase class. This class creates an instance of GripperController that serves as the controller for the arm’s end effector. This class also contains the code for inverse kinematics abstraction. 

**widowx_env:** Initializes an environment for interacting with the WidowX arm. This class inherits from the RobotBaseEnv class, creates an instance of WidowXController, and provides additional functionality for starting up and resetting the robot.

**widowx_env_service:** Creates WidowXActionServer and WidowXClient classes, which allow Python scripts to interface with the robot by running an instance of WidowXActionServer and then creating an instance of WidowXClient in a separate script.

**pick_and_place:** Creates a function for picking up an object at a certain location and placing it at another location using commands in the WidowXClient class.

**chess_ai:** This is a submodule that contains the chess engine that the bot uses to calculate its next move, which is located in the file minimax.py.

**bot:** This file is the executable that reads the user’s moves, calculates the next move, and plays that move on the physical board.


In this implantation we could dig more into the machine vision aspect of the project: not only analyzing with machine vision but also finding more elegant, and sensor based methods of localization.

As we go into the next week Mia and Dan will be working on the GUI Eddie will be working on the pose array, Kate will continue working with April tags and and Will will be working on the literal edge cases of the chess words. We plan to have our MVP completed by Wednesday so that we have time to give some of our undivided attention and work on our website.
