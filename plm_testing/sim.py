"""Simulation for adaptive path planning"""

import random

from matplotlib import pyplot as plt
import numpy as np

from plm_testing.a_star_octree import calculate_h, find_path, find_unknown_leaves, is_safe_node
from plm_testing.octree import OctreeNode

# Config
# Simulation Parameters
NUM_OBSTACLES = 200
START_POSITION = (-18, -18, -18)
WORLD_BOUNDS = 200.0

# Sensor Parameters
SENSOR_RANGE = 10.0
SENSOR_MIN_RANGE = 0.3
CAMERA_FOV_H = 87.0
CAMERA_FOV_V = 58.0
CAMERA_RES_H = 40
CAMERA_RES_V = 27

# Initial Exploration
INIT_RADIUS = 10.0
INIT_CIRCLE_STEPS = 40

# Visualization
VISUALIZATION = True
STEP_DELAY = 0.00005
VIZ_FIGURE_SIZE = (10, 8)

# Display Alpha
ALPHA_FRONTIER = 0.8            # Opacity of frontier markers
ALPHA_OBSTACLE = 0.1            # Opacity of true obstacles (ghost)

def visualize(ax, octree, robot_pos, path, frontiers, true_obstacles, title_text):
    """Helper to keep the main loop clean."""
    if not VISUALIZATION: return
    ax.clear()
    octree.draw(ax)
    
    # Draw Path
    if path:
        # Check if path is list of Nodes or Coordinates
        try:
            xs, ys, zs = zip(*[n.position for n in path])
        except AttributeError:
            xs, ys, zs = zip(*path) # Handle raw coordinate tuples
        ax.plot(xs, ys, zs, color='blue', linewidth=2)

    # --- 1. Draw Sensor Range (Wireframe Sphere) --- ####llm
    # Create a generic sphere
    u = np.linspace(0, 2 * np.pi, 13) # Low resolution for speed (13 points)
    v = np.linspace(0, np.pi, 7)
    x = SENSOR_RANGE * np.outer(np.cos(u), np.sin(v)) + robot_pos[0]
    y = SENSOR_RANGE * np.outer(np.sin(u), np.sin(v)) + robot_pos[1]
    z = SENSOR_RANGE * np.outer(np.ones(np.size(u)), np.cos(v)) + robot_pos[2]
    
    # Plot as a light wireframe
    ax.plot_wireframe(x, y, z, color='cyan', alpha=0.2)
    # -----------------------------------------------

    # Draw Robot
    ax.scatter([robot_pos[0]], [robot_pos[1]], [robot_pos[2]], color='green', s=100, label='Robot')

    # Draw Frontiers (Make them visible!)
    if frontiers:
        fx, fy, fz = zip(*[f.position for f in frontiers])
        ax.scatter(fx, fy, fz, c='orange', s=30, alpha=0.8, label='Frontiers') # Increased size/alpha

    # Draw True Obstacles (Ghost)
    obs_x, obs_y, obs_z = zip(*true_obstacles)
    ax.scatter(obs_x, obs_y, obs_z, c='black', alpha=0.1, s=10)
    
    ax.set_xlim(-WORLD_BOUNDS, WORLD_BOUNDS); ax.set_ylim(-WORLD_BOUNDS, WORLD_BOUNDS); ax.set_zlim(-WORLD_BOUNDS, WORLD_BOUNDS)
    ax.set_title(title_text)
    plt.draw()
    plt.pause(STEP_DELAY)

def update_cell_with_max_size(root, pos, is_occupied, max_leaf_size=2.0):
    """
    Updates a cell, but forces a split if the current leaf is larger than max_leaf_size.
    This ensures we don't accidentally mark a huge chunk of the map as 'known' 
    just because we saw one tiny point.
    """
    # 1. Find the leaf at this position
    leaf = root.find_leaf(pos)
    
    # 2. If the leaf is huge (radius > max_leaf_size), split it!
    # We loop until it's small enough.
    while leaf.radius > max_leaf_size:
        leaf.splitting() #
        # After splitting, find the specific child that contains 'pos'
        leaf = root.find_leaf(pos)

    # 3. Now that we are at a small enough resolution, update the status
    if is_occupied:
        leaf.status = 'occupied'
    else:
        leaf.status = 'empty'

    # 4. Cleanup parents (optional, but keeps tree clean)
    leaf.update_parents() #

def simulate_sensor(robot_pos, true_obstacles, octree_root, sensor_range=SENSOR_RANGE, sensor_min_range=SENSOR_MIN_RANGE): # trajectory, 
    """
    Simulates a sensor reading by taking a sphere around the robot position and assuming all objects within range are detected.
    
    1. Detects obstacles within sensor_range.
    2. Clears space (marks empty) along the line of sight with raycast (simplified as a radius clear).

    Limitations:
    - Senses everyting within range (even if behind obstacle, which is different from true camera sensor)
    - Projects as sphere, so the edge of detection is different then the effectively adaptive camera "edge"
    
    returns: True if the map changed, False otherwise.
    """
    map_changed = False
    rand_sphere_surf_points = []

    # Set all nodes in scan sphere to be empty
    
    # rand_sphere_surf_points = Generate random "end" positions at sphere surface:
    for i in range(200):
        theta = random.random() * 2 * np.pi
        phi = random.random() * 2 * np.pi
        rho = sensor_range
        x = rho * np.sin(phi) * np.cos(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        act_x = robot_pos[0] + x
        act_y = robot_pos[0] + y
        act_z = robot_pos[0] + z
        rand_sphere_surf_points[i] = (act_x, act_y, act_z)
    # scan_sphere_sampled = raycast to those end positions (rand_sphere_surf_points)
    for point in rand_sphere_surf_points:
        octree_root.raycast(origin=robot_pos, target=point)
    
    
    # Detect Obstacles (overwrite previous empty assumption)
    for obs in true_obstacles:
        dist = np.sqrt((obs[0]-robot_pos[0])**2 + (obs[1]-robot_pos[1])**2 + (obs[2]-robot_pos[2])**2)
        
        if dist <= sensor_range and dist > sensor_min_range:
            # Skip obstacles outside map bounds
            leaf = octree_root.find_leaf(obs)
            if leaf is None:
                continue

            if leaf.status != 'occupied':
                update_cell_with_max_size(octree_root, obs, is_occupied=True, max_leaf_size=1.0)
                map_changed = True

            octree_root.raycast(origin=robot_pos, target=obs) # does this take care of what the scan_sphere empty thing? I think this misses the nodes that don't lead to an occupied node

    # Affirm not occupied
    current_node = octree_root.find_leaf(obs)
    if current_node and current_node.status != 'empty':
        update_cell_with_max_size(octree_root, robot_pos, is_occupied=False, max_leaf_size=1.0)
        # map_changed = True

    # Clear Empty Space ###?
    return map_changed

def autonomous_exploration(start_pos, true_obstacles):
    """
    Order of exploration:
    1. Circle
    - Move to circle point
    - Update with sensor
    - repeat until circle completed
    2. Adaptive path planning
    - Move to nearest unknown update position
    - Scan
    - repeat until circle completed
    3. Ending procedure [messaging]
    """

    ## Setup
    spatial_map = OctreeNode(position=(0,0,0), radius=WORLD_BOUNDS) 
    current_pos = start_pos
    
    # Setup Visualization
    if VISUALIZATION:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    print(f"--- Starting Exploration at {start_pos} ---")
    
    ## Circle time
    # Generate Circle Waypoints
    circle_steps = 10
    angles = np.linspace(0, 2*np.pi, circle_steps)
    circle_path = []
    for theta in angles:
        x = INIT_RADIUS * np.cos(theta)
        y = INIT_RADIUS * np.sin(theta)
        z = 0.03
        circle_path.append((x, y, z))
    
    circle_start_target = circle_path[0]

    print("Circle time")
    
    # Plan path to the first point of the circle
    transit_path = find_path(spatial_map, current_pos, circle_start_target)
    
    if transit_path:
        for goal_node in transit_path:
            current_pos = goal_node.position
            simulate_sensor(current_pos, true_obstacles, spatial_map)
            visualize(ax, spatial_map, current_pos, transit_path, [], true_obstacles, "Phase 0: Transit to Circle")
    else:
        print("Teleport")
        current_pos = circle_start_target

    print("Phase 1: Performing Circular Scan...")
    
    # Circle itself
    for i, pos in enumerate(circle_path):
        current_pos = pos
        simulate_sensor(current_pos, true_obstacles, spatial_map)
        visualize(ax, spatial_map, current_pos, circle_path[i:], [], true_obstacles, "Phase 1: Circular Scan")

    ## Adaptive pathing
    iteration = 0
    while True:
        iteration += 1
        
        #### LLM Debug
        if iteration % 10 == 0:
            print(f"\n[Debug] Iter {iteration}: {len(frontiers)} frontiers, exploring {goal_pos}")

        # 1. Find Frontiers
        frontiers = []
        find_unknown_leaves(spatial_map, frontiers) #
        
        if not frontiers:
            print(f"\n[Iter {iteration}] Exploration Complete: No unknown leaves left.")
            break
        
        # 2. Choose Goal
        goal_leaf = min(frontiers, key=lambda leaf: calculate_h(leaf.position, current_pos)) #
        goal_pos = goal_leaf.position
        
        # Terminal Status Update
        print(f"\r[Iter {iteration}] Pos: {current_pos} | Goal: {goal_pos} | Unk Leaves: {len(frontiers)}", end="")

        # 3. Plan Path
        path = find_path(spatial_map, current_pos, goal_pos) #

        if not path:
            print("\n  -> Goal Unreachable. Marking occupied.")
            spatial_map.update_cell(goal_pos, is_occupied=True)
            continue
        
        # 4. Execute Path
        for i, node in enumerate(path):
            current_pos = node.position

            if i < len(path) - 1:
                next_node = path[i + 1]
                if not is_safe_node(next_node, spatial_map):
                    print("\n  -> Path blocked ahead! Replanning...")
                    break

            # Simulate Sensor
            changed = simulate_sensor(current_pos, true_obstacles, spatial_map)
            
            visualize(ax, spatial_map, current_pos, path[i:], frontiers, true_obstacles, f"Phase 2: Adaptive | Iter {iteration}")

            if changed:
                print("\n  -> Map Changed! Replanning...")
                break 

    print("\nMission Finished.")
    if VISUALIZATION:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # Define ground truth
    true_obstacles = []
    random.seed(1)  # For reproducibility
    for i in range(NUM_OBSTACLES):
        true_obstacles.append((
            random.uniform(-WORLD_BOUNDS+5, WORLD_BOUNDS-5),
            random.uniform(-WORLD_BOUNDS+5, WORLD_BOUNDS-5),
            random.uniform(-WORLD_BOUNDS+5, WORLD_BOUNDS-5)
        ))

    autonomous_exploration(START_POSITION, true_obstacles)