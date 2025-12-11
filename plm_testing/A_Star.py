"""Implement a 3D A* pathfinding algorithm for a robotic arm to navigate"""

import heapq
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Define the Node class
class Node:
    def __init__(self, position: Tuple[int, int, int], g: float = float('inf'), 
                 h: float = 0.0, parent: "Node" = None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

# Check if a cell is valid
def is_valid(grid, row, col, depth):
    ROW, COL, DEPTH = grid.shape
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL) and (depth >= 0) and (depth < DEPTH)

# Check if a cell is unblocked
def is_unblocked(grid, row, col, depth):
    return grid[row, col, depth] == 0

# Euclidian distance
def calculate_h(position, dest):
    h = np.sqrt((position[0] - dest[0])**2 + (position[1] - dest[1])**2 + (position[2] - dest[2])**2)
    return float(h)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    Get all valid neighboring positions in the grid.
    
    Args:
        grid: 2D numpy array where 0 represents walkable cells and 1 represents obstacles
        position: Current position (x, y)
    
    Returns:
        List of valid neighboring positions
    """
    x, y, z = position
    
    # Set directional moves
    possible_moves = [
        (x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x+1, y+1, z), (x-1, y-1, z), (x+1, y-1, z), (x-1, y+1, z),
        (x, y, z+1), (x, y, z-1), (x+1, y, z+1), (x-1, y, z-1), (x, y+1, z+1), (x, y-1, z-1), (x+1, y+1, z+1), 
        (x-1, y-1, z-1), (x+1, y-1, z+1), (x-1, y+1, z-1), (x+1, y, z-1), (x-1, y, z+1), (x, y+1, z-1), 
        (x, y-1, z+1), (x+1, y-1, z-1), (x-1, y+1, z+1), (x-1,y-1,z+1), (x+1,y+1,z-1)
    ]

    neighbors = []
        
    for nx, ny, nz in possible_moves:
        if is_valid(grid, nx, ny, nz):
            if is_unblocked(grid, nx, ny, nz):
                neighbors.append((nx, ny, nz))

    return neighbors

def reconstruct_path(goal_node: Node) -> List[Tuple[int, int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current.position)
        current = current.parent
        
    return path[::-1]  # Reverse to get path from start to goal

def find_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
    
    Args:
        grid: 3D numpy array (0 = free space, 1 = obstacle)
        start: Starting position (x, y, z)
        goal: Goal position (x, y, z)
    
    Returns:
        List of positions representing the optimal path
    """

    # Initialize start node
    start_node = Node(
        position=start,
        g=0,
        h=calculate_h(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node.f, start)]  # Priority queue
    open_dict = {start: start_node}         # For quick node lookup
    closed_set = []                      # Explored nodes
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if at the goal
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.append(current_pos)
        
        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost
            tentative_g = current_node.g + calculate_h(current_pos, neighbor_pos)
            
            # Create or update neighbor
            if neighbor_pos not in open_dict:
                neighbor = Node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_h(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor.f, neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos].g:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor.g = tentative_g
                neighbor.f = tentative_g + neighbor.h
                neighbor.parent = current_node
    
    return []  # No path found
    
def visualize_path(grid: np.ndarray, path: List[Tuple[int, int, int]]):
    """
    Visualize the 3D grid and found path.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Obstacles
    # Get coordinates where grid is 1
    obs_x, obs_y, obs_z = np.where(grid == 1)
    ax.scatter(obs_x, obs_y, obs_z, c='black', marker='s', s=100, alpha=0.3, label='Obstacles')
    
    # 2. Plot Path
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=4, label='Path')
        
        # Start (Green) and Goal (Red)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='green', s=200, label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='red', s=200, label='Goal')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    plt.title("3D A*")
    plt.show()

def main():
    # Empty occupancy grid
    grid = np.zeros((20, 20, 20), dtype=int)  # 20x20 grid

    # Obstacles
    grid[5:15, 2:10, 10:20] = 1 
    grid[5:20, 5:15, 10:20] = 1
    grid[10:12, 0:15, 0:15] = 1

    # Define start and goal positions
    start_pos = (2, 2, 2)
    goal_pos = (18, 18, 18)

    # Find the path
    path = find_path(grid, start_pos, goal_pos)
    if path:
        print(f"Path found with {len(path)} steps!")
        visualize_path(grid, path)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()