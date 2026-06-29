"""Frontier navigation using A* on an Octree"""

import heapq
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from plm_testing.octree import OctreeNode
import random

#### LLM
SAFETY_MARGIN = 0.02

# Functions
def calculate_h(position, dest):
    """Calculates the Euclidean distance heuristic between two points.

    Args:
        position: Current (x, y, z) position.
        dest: Goal (x, y, z) position.

    Returns:
        float: The Euclidean distance.
    """
    h = np.sqrt((position[0] - dest[0])**2 + (position[1] - dest[1])**2 + (position[2] - dest[2])**2)
    return float(h)

def reconstruct_path(node_path, current_node): # Consider changing only the region around the new information? do I already do that?
    """
    Reconstruct the path from goal to start.
    """
    path = [current_node]
    while current_node in node_path:
        current_node = node_path[current_node]
        path.append(current_node)
    return path[::-1]

def is_safe_node(node: OctreeNode, root: OctreeNode, safety_margin: float = SAFETY_MARGIN) -> bool: ####LLM
    """
    Check if a node is safe to traverse (not occupied and maintains safety margin).
    A node is unsafe if it's occupied OR if an occupied cell is within safety_margin.
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

def find_path(root: OctreeNode, start, goal) -> List[OctreeNode]:
    """Implements A* to find path from start to goal through Octree positions

    Args:
        root: The root of the Octree.
        start: The (x, y, z) start coordinates.
        goal: The (x, y, z) goal coordinates.

    Returns:
        List[OctreeNode]: A list of nodes representing the path from start to goal.
                          Returns an empty list [] if no path is found.
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

def closest_unknown(root: OctreeNode, start_pos: Tuple[float, float, float]) -> Tuple[float, float, float]: ### Consider changing to return the node itself to match other method conventions
    """Identify position of closest unknown cell
    
    Args:
        root: The Octree root node.
        start_pos: The robot's current position.
    
    Returns:
        None: if no unknown leaves
        Tuple[x,y,z]: the position of the closest unknown Octree node
        """
    unknown_leaves = []
    find_unknown_leaves(root, unknown_leaves)

    if not unknown_leaves:
        return None

    closest_unknown = min(unknown_leaves, key=lambda leaf: calculate_h(leaf.position, start_pos))
    return closest_unknown.position

def go_to_goal(root: OctreeNode, start_pos: Tuple[float, float, float]) -> List[OctreeNode]:
    """Identifies the nearest unknown frontier and plans an A* path to it.
    
    If a path cannot be found to the closest unknown node (e.g., it is unreachable),
    that node is marked as 'occupied' to prevent infinite loops, and the next 
    closest frontier is selected.

    Args:
        root: The Octree root node.
        start_pos: The robot's current position.

    Returns:
        List[OctreeNode]: The planned path, or [] if no frontiers remain.
    """
    standoff_distance = 0.08

    while True:
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

        path = find_path(root, start_pos, goal_pos)
        if path:
            return path
        
        # Failure case. Assumes an unreachable node is surrounded by occupied and thus for scanning purposes, also occupied
        root.update_cell(unknown_pos, is_occupied=True)


def find_unknown_leaves(node: OctreeNode, unknown_leaves: list):
    """Collects all leaf nodes with status 'unknown'. Suggested use is with root node for all unknowns in root's space.
    
    Args:
        node: The current node to check.
        unknown_leaves: A list to append found unknown nodes to (modified in-place).
    """
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

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()


if __name__ == "__main__":
    root = OctreeNode(position=(0,0,0), radius=0.8, min_radius=0.02)

    # Add obstacles
    for _ in range(15):
        obs = (random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), random.uniform(-0.5,0.5))
        root.insert_obstacle(obs)

    #### LLM
    # Wall 1: A vertical barrier at X=0 (Blocks the direct center)
    # Spans Y: -10 to 10, Z: -10 to 10
    for y in np.arange(-0.4, 0.4, 0.04): # 0.04 is the step size (4cm)
        for z in np.arange(-0.4, 0.4, 0.04):
            root.insert_obstacle((0, y, z))

    # Wall 2: A floor/ceiling plate at Z=-0.4 (Forces the robot to go up)
    # Spans X: -10 to 10, Y: -10 to 10
    for x in np.arange(-0.4, 0.4):
        for y in np.arange(-0.4, 0.4):
            root.insert_obstacle((x, y, -5))

    # Wall 3: A side barrier at Y=8 (Narrowing the path near the goal)
    # Spans X: 5 to 15, Z: 5 to 15
    for x in np.arange(5, 16):
        for z in np.arange(5, 16):
            root.insert_obstacle((x, 8, z))
    ####

    start_pos = (-0.4, -0.4, -0.4)
    goal_pos = (0.4, 0.4, 0.4)

    path = find_path(root, start_pos, goal_pos)
    if path:
        print(f"Found path with {len(path)} nodes")
        visualize_path(root, path)
    else:
        print("No path found")