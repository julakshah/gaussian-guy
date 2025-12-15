"""Frontier navigation using A* on an Octree"""
"""
While true:
    frontiers = find_unknown_leaves(octree)

    if frontiers is empty:
        break
    
    goal = closest_frontier(frontiers, start_pos)
    path = A_Star(octree, start_pos, goal)

    if path is empty:
        set goal as occupied
        continue
    
    for node in path:
        move_to(node.position)

        sensor_data = get_sensor_data(current_pos)
"""

import heapq
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from Octree import OctreeNode
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

#### LLM
SAFETY_MARGIN = 0.5

def calculate_h(position, dest):
    h = np.sqrt((position[0] - dest[0])**2 + (position[1] - dest[1])**2 + (position[2] - dest[2])**2)
    return float(h)

def reconstruct_path(node_path, current_node):
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = [current_node]
    while current_node in node_path:
        current_node = node_path[current_node]
        path.append(current_node)
    return path[::-1]

def is_safe_node(node: OctreeNode, root: OctreeNode, safety_margin: float = SAFETY_MARGIN) -> bool: ####LLM
    """
    Check if a node is safe to traverse (not occupied and maintains safety margin).
    A node is unsafe if it's occupied OR if any occupied cell is within safety_margin.
    """
    if node.status == 'occupied':
        return False
    
    # Check all neighbors within safety margin
    # We need to check a sphere of radius safety_margin around the node
    neighbors = node.get_neighbors(root)
    for neighbor in neighbors:
        if neighbor.status == 'occupied':
            dist = calculate_h(node.position, neighbor.position)
            if dist < safety_margin:
                return False
    
    return True

def find_path(start, goal, root: OctreeNode) -> List[Tuple[int, int]]:
    """
    A* algorithm
    """
    # Branch octree to get start and goal nodes
    start_node = root.find_leaf(start)
    goal_node = root.find_leaf(goal)
    
    if start_node is None or goal_node is None:
        print("Start or Goal is out of bounds!")
        return []
    
    queue = [(0, 0, start_node)] # (f_score, tie_breaker, node)
    node_path = {}    # Tracks the path:  child -> parent
    g_score = {start_node: 0}
    f_score = {start_node: calculate_h(start_node.position, goal_node.position)}                  
    
    step_count = 0
    closed_set = set()

    while queue:
        _, _, current_node = heapq.heappop(queue)
        
        if current_node in closed_set:
            continue
        closed_set.add(current_node)

        if current_node == goal_node:
            return reconstruct_path(node_path, current_node)
        
        for neighbor in current_node.get_neighbors(root):
            if not is_safe_node(neighbor, root):
                continue

            tentative_g = g_score[current_node] + calculate_h(current_node.position, neighbor.position)

            if tentative_g < g_score.get(neighbor, float('inf')):
                node_path[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + calculate_h(neighbor.position, goal_node.position)
                
                step_count += 1
                heapq.heappush(queue, (f_score[neighbor], step_count, neighbor))
        
    return []

def closest_unknown(root: OctreeNode, start_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Identify closest unknown cell"""
    unknown_leaves = []
    find_unknown_leaves(root, unknown_leaves)

    if not unknown_leaves:
        return None

    closest_unknown = min(unknown_leaves, key=lambda leaf: calculate_h(leaf.position, start_pos))
    return closest_unknown.position

def go_to_goal(root: OctreeNode, start_pos: Tuple[float, float, float]) -> List[Tuple[int, int]]:
    """Plan path to goal using A*"""
    standoff_distance = 2.5

    unknown_pos = closest_unknown(root, start_pos)

    if unknown_pos is None:
        return []
    
    start_arr = np.array(start_pos)
    unknown_arr = np.array(unknown_pos)

    direction = start_arr - unknown_arr

    # Calculate distance
    dist = np.linalg.norm(direction)

    if dist < 1e-6:
        # Default to backing up in Z+ direction (or any safe default)
        normalized_dir = np.array([0, 0, 1])
    else:
        normalized_dir = direction / dist

    goal_pos = tuple(unknown_arr + (normalized_dir * standoff_distance))

    path = find_path(start_pos, goal_pos, root)
    return path

def find_unknown_leaves(node: OctreeNode, unknown_leaves: list):
    # If leaf node
    if node.children[0] is None:
        if node.status == "unknown":
            unknown_leaves.append(node)
        return
    
    # Else check all children
    for child in node.children:
        find_unknown_leaves(child, unknown_leaves)

def visualize_path(root: OctreeNode, path: List[OctreeNode]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    root.draw(ax)

    if path:
        xs, ys, zs = zip(*[node.position for node in path])
        ax.plot(xs, ys, zs, color='blue', linewidth=2, label='Path')
        ax.scatter(xs[0], ys[0], zs[0], color='green', s=100, label='Start')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=100, label='Goal')

    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()


if __name__ == "__main__":
    root = OctreeNode(position=(0,0,0), r=20)

    # Add obstacles
    for _ in range(15):
        obs = (random.uniform(-20,20), random.uniform(-20,20), random.uniform(-20,20))
        root.insert_obstacle(obs)

    #### LLM
    # Wall 1: A vertical barrier at X=0 (Blocks the direct center)
    # Spans Y: -10 to 10, Z: -10 to 10
    for y in range(-10, 11):
        for z in range(-10, 11):
            root.insert_obstacle((0, y, z))

    # Wall 2: A floor/ceiling plate at Z=-5 (Forces the robot to go up)
    # Spans X: -10 to 10, Y: -10 to 10
    for x in range(-10, 11):
        for y in range(-10, 11):
            root.insert_obstacle((x, y, -5))

    # Wall 3: A side barrier at Y=8 (Narrowing the path near the goal)
    # Spans X: 5 to 15, Z: 5 to 15
    for x in range(5, 16):
        for z in range(5, 16):
            root.insert_obstacle((x, 8, z))
    ####

    start_pos = (-15, -15, -15)
    goal_pos = (15, 15, 15)

    path = find_path(start_pos, goal_pos, root)
    if path:
        print(f"Found path with {len(path)} nodes")
        visualize_path(root, path)
    else:
        print("No path found")