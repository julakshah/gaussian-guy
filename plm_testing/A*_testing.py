import heapq
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Removed create_node function and unused Cell class
# Implemented Node class to replace dictionary structure
class Node:
    def __init__(self, position: Tuple[int, int], g: float = float('inf'), 
                 h: float = 0.0, parent: 'Node' = None):
        """
        Initialize a node for the A* algorithm.
        
        Args:
            position: (x, y) coordinates of the node
            g: Cost from start to this node
            h: Estimated cost from this node to goal
            parent: Parent node
        """
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

# Define the size of the grid
ROW = 9
COL = 10

def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def is_unblocked(grid, row, col):
    return grid[row][col] == 1

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h(position, dest):
    h = np.sqrt((position[0] - dest[0])**2 + (position[1] - dest[1])**2 )
    return float(h)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all valid neighboring positions in the grid.
    """
    x, y = position
    rows, cols = grid.shape
    
    possible_moves = [
        (x+1, y), (x-1, y),    # Right, Left
        (x, y+1), (x, y-1),    # Up, Down
        (x+1, y+1), (x-1, y-1),  # Diagonal moves
        (x+1, y-1), (x-1, y+1)
    ]
    
    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols
        and grid[nx, ny] == 0
    ]

def reconstruct_path(goal_node: Node) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current.position) # Accessed via attribute
        current = current.parent      # Accessed via attribute
        
    return path[::-1]

def find_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
    """

    # Initialize start node using Class
    start_node = Node(
        position=start,
        g=0,
        h=calculate_h(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node.f, start)]  # Priority queue: Accessed via attribute
    open_dict = {start: start_node}      # For quick node lookup
    closed_set = []                      
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.append(current_pos)
        
        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost (Accessed via attribute)
            tentative_g = current_node.g + calculate_h(current_pos, neighbor_pos)
            
            # Create or update neighbor
            if neighbor_pos not in open_dict:
                # Instantiate new Node object
                neighbor = Node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_h(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor.f, neighbor_pos)) # Accessed via attribute
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos].g: # Accessed via attribute
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor.g = tentative_g
                neighbor.f = tentative_g + neighbor.h
                neighbor.parent = current_node
    
    return []

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]):
    """
    Visualize the grid and found path.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')
    
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("A* Pathfinding Result")
    plt.show()

def main():
    # Create a sample grid
    grid = np.zeros((20, 20), dtype=int)
    # Add some obstacles
    grid[5:15, 10] = 1
    grid[5, 5:15] = 1
    # Define start and goal positions
    start_pos = (2, 2)
    goal_pos = (18, 18)
    # Find the path
    path = find_path(grid, start_pos, goal_pos)
    if path:
        print(f"Path found with {len(path)} steps!")
        visualize_path(grid, path)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()