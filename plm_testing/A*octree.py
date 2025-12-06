import heapq
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from Octree import OctreeNode
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time


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

def find_path(start, goal, root: OctreeNode) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
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

            if neighbor.status == 'occupied':
                continue

            tentative_g = g_score[current_node] + calculate_h(current_node.position, neighbor.position)

            if tentative_g < g_score.get(neighbor, float('inf')):
                node_path[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + calculate_h(neighbor.position, goal_node.position)
                
                step_count += 1
                heapq.heappush(queue, (f_score[neighbor], step_count, neighbor))
        
    return []

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
