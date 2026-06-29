"""Implement a custom Octree class for use in Gaussian Guy CompRobo final"""
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class OctreeNode:
    """
    Custom Octree node with an assigned status. 

    Attributes:
        position: Center of node in (x, y, z) (world frame)
        radius: Closest distance from position to any face
        status:
            One of {'unknown', 'empty', 'occupied', 'container'}.

            - 'unknown':
                Space has not been observed yet.

            - 'empty':
                Space has been observed and confirmed free.

            - 'occupied':
                Space contains a solid obstacle.

            - 'container':
                Internal node that has been subdivided into children.
                The node itself does not represent a single voxel;
                its children hold the actual occupancy information.
        parent: Reference to the parent OctreeNode (container) or none if root
        children: List of 8 child OctreeNodes or None if leaf
        min_radius: Minimum radius before stopping subdivision

    """
    def __init__(self, position, radius=1, parent=None, min_radius=0.02):
        self.position = position    # (x, y, z)
        self.radius = radius  # radius
        self.status = 'unknown'
        self.parent = parent
        self.children = [None] * 8  # Child nodes
        self.min_radius = min_radius

    def splitting(self):
        """Subdivides self (OctreeNode) into 8 equal-sized child nodes. Changes the node's status to 'container'.
        """        
        # Check if already split or minimum size
        if (self.children[0] is not None) or (self.radius / 2 < self.min_radius):
            return
        
        # Values for child nodes
        new_r = self.radius / 2

        child_positions = [] # positives are right, up, front
        for xi in (-1, 1):
            for yi in (-1, 1):
                for zi in (-1, 1):
                    child_positions.append((xi, yi, zi))


        # Create child nodes
        for i, pos in enumerate(child_positions):
            self.children[i] = OctreeNode(
                position=(
                    self.position[0] + pos[0] * new_r,
                    self.position[1] + pos[1] * new_r,
                    self.position[2] + pos[2] * new_r,
                ),
                radius=new_r,
                parent=self,
                min_radius=self.min_radius,
            )
        self.status = 'container'
    
    def get_status(self):
        """Determines if node is container and then returns the status of the node.
        
        Returns:
            str: 'container' if the node has children, otherwise returns the status ('occupied', 
            'empty', or 'unknown').
        """
        # If children, container node so status is irrelevant.
        if self.children[0] is not None:
            return 'container' 
        
        # If no children, return actual data
        return self.status
    
    def prune(self):
        """Removes unnecessary child nodes to save memory
        
        If all 8 children are leaves and share the same status, they are deleted and the parent is assigned that status.
        """
        if self.children[0] is None:
            return
            
        # Only prune if all children are leaves #### LLM
        if any(child.children[0] is not None for child in self.children):
            return
        
        # Check if all children have same status
        first_status = self.children[0].status
        if all(child.status == first_status for child in self.children):
            self.status = first_status
            self.children = [None] * 8  # Remove children

    def insert_obstacle(self, obstacle_pos): ### change to be "insert_entity" and pass in which type of node
        """Recursively traverses the tree to find the leaf node containing the
        obstacle and assign status. If the leaf is larger than min_radius, it splits.

        Args:
            obstacle_pos: A tuple (x, y, z) representing the obstacle's location.

        Returns:
            bool: True if the obstacle was successfully inserted. 
                  False if the obstacle was out of the node's bounds.
        """        
        current_node = self

        # Check if obstacle in node bounds
        if (abs(obstacle_pos[0] - self.position[0]) > self.radius or
            abs(obstacle_pos[1] - self.position[1]) > self.radius or
            abs(obstacle_pos[2] - self.position[2]) > self.radius):
            return False 

        # Check if already occupied
        if self.get_status() == 'occupied':
            return True
        
        # If at minimum size, mark as occupied
        while True:

            if current_node.children[0] is None:
                current_node.splitting()

                if current_node.children[0] is None:
                    break  # Reached min size
            
            # Check child nodes
            relative_pos = (
                obstacle_pos[0] - current_node.position[0],
                obstacle_pos[1] - current_node.position[1],
                obstacle_pos[2] - current_node.position[2],
            )

            if relative_pos[0] > 0:
                x_index = 1
            else:
                x_index = 0
            if relative_pos[1] > 0:
                y_index = 1
            else:
                y_index = 0
            if relative_pos[2] > 0:
                z_index = 1
            else:
                z_index = 0

            child_index = x_index * 4 + y_index * 2 + z_index
            current_node = current_node.children[child_index]

        current_node.status = "occupied"
        current_node.update_parents()
        return True
    
    def find_leaf(self, position):
        """Traverses the tree to find the specific leaf node containing a point.

        Args:
            position: A tuple (x, y, z) in world frame.

        Returns:
            OctreeNode: The leaf node containing the position.
            None: If the position is outside the root node's bounds.
        """
        current_node = self
        
        # Check if in bounds
        if (abs(position[0] - current_node.position[0]) > current_node.radius or
            abs(position[1] - current_node.position[1]) > current_node.radius or
            abs(position[2] - current_node.position[2]) > current_node.radius):
            return None

        # Go down each child node
        while current_node.children[0] is not None:
            
            # Check child nodes
            relative_pos = (
                position[0] - current_node.position[0],
                position[1] - current_node.position[1],
                position[2] - current_node.position[2],
            )

            if relative_pos[0] > 0:
                x_index = 1
            else:
                x_index = 0
            if relative_pos[1] > 0:
                y_index = 1
            else:
                y_index = 0
            if relative_pos[2] > 0:
                z_index = 1
            else:
                z_index = 0

            child_index = x_index * 4 + y_index * 2 + z_index
                        
            # Move down
            current_node = current_node.children[child_index]
            
        # Leaf node exit
        return current_node

    def update_cell(self, pos, is_occupied):
        """Updates the status of a specific coordinate.

        Args:
            pos: The (x, y, z) coordinate to update.
            is_occupied: True to mark 'occupied', False to mark 'empty'.
        """
        leaf = self.find_leaf(pos)
        if leaf is None:
            return

        if is_occupied:
            leaf.status = "occupied"
        else:
            leaf.status = "empty"

        leaf.update_parents()

    def update_parents(self):
        """Update parent nodes to prune unnecessary children"""
        current_node = self
        while current_node.parent is not None:
            current_node.parent.prune()
            current_node = current_node.parent

    def get_neighbors(self, root) -> list:
        """Checks for leaf nodes adjacent to the current node.

        Checks 26 discrete directions (faces, edges, corners) around the node and returns the set of nodes from those directions
        
        Args:
            root (OctreeNode): The root of the tree, required to search for 
                               neighbors that may exist in different branches.

        Returns:
            list[OctreeNode]: A list of unique adjacent leaf nodes.
        """
        neighbors = set()

        directions = [] # positives are right, up, front
        for xi in (-1, 0, 1):
            for yi in (-1, 0, 1):
                for zi in (-1, 0, 1):
                    if (xi, yi, zi) != (0, 0, 0):
                        directions.append((xi, yi, zi))

        for direction in directions:
            epsilon = 1e-6

            neighbor_pos = (
                self.position[0] + direction[0] * (self.radius + epsilon),
                self.position[1] + direction[1] * (self.radius + epsilon),
                self.position[2] + direction[2] * (self.radius + epsilon),
            )
            neighbor_node = root.find_leaf(neighbor_pos)
            if neighbor_node:
                neighbors.add(neighbor_node)

        return list(neighbors)
    
    def raycast(self, origin, target):
        """Uses a step-based traversal to mark all voxels along the vector from 
        origin to target as 'empty'. ### Consider changing to pass in what to raycast the thing is, or simply return a list of the nodes vs assuming empty?

        Args:
            origin: The (x, y, z) starting point of the ray (e.g., camera position).
            target: The (x, y, z) end point of the ray (e.g., detected obstacle).
        """        
        diff = (
            target[0] - origin[0],
            target[1] - origin[1],
            target[2] - origin[2],
        )

        total_dist = (diff[0]**2 + diff[1]**2 + diff[2]**2) ** 0.5
        
        # Safety check for zero distance
        if total_dist == 0:
            return

        # Normalize direction (The vector of length 1 pointing to target)
        direction = (diff[0] / total_dist, diff[1] / total_dist, diff[2] / total_dist)
        
        # 2. Step size
        # radius/2 ensures we don't skip over any voxels (Nyquist sampling)
        step_size = self.min_radius / 2 ### Change to leaf size?
        
        # 3. Loop using simple addition
        current_dist = 0.0
        current_pos = list(origin) # Make mutable
        
        # STOP condition: Stop 1 step before the end so we don't erase the target
        while current_dist < (total_dist - step_size):
            
            leaf = self.find_leaf(current_pos)
            
            # Only mark if it's currently unknown or occupied? 
            # Usually we overwrite occupied obstacles if we see through them (dynamic)
            # But for static scanning, checking "!= occupied" is safer.
            if leaf and leaf.status != 'occupied':
                leaf.status = 'empty'
            
            # Move forward by adding the step vector
            current_pos[0] += direction[0] * step_size
            current_pos[1] += direction[1] * step_size
            current_pos[2] += direction[2] * step_size
            
            current_dist += step_size
        

    def draw_cube(self, ax, facecolor='red', edgecolor='black', alpha=1.0): ####
        """Draws a cube at this node."""
        x, y, z = self.position
        radius = self.radius
        corners = [
            (x-radius, y-radius, z-radius), (x-radius, y-radius, z+radius), (x-radius, y+radius, z-radius), (x-radius, y+radius, z+radius),
            (x+radius, y-radius, z-radius), (x+radius, y-radius, z+radius), (x+radius, y+radius, z-radius), (x+radius, y+radius, z+radius),
        ]
        faces = [
            [corners[i] for i in [0,1,3,2]],
            [corners[i] for i in [4,5,7,6]],
            [corners[i] for i in [0,1,5,4]],
            [corners[i] for i in [2,3,7,6]],
            [corners[i] for i in [0,2,6,4]],
            [corners[i] for i in [1,3,7,5]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=0.5, alpha=alpha))

    def draw(self, ax): ####
        """Recursively draws the node and its children on a 3D plot. Renders container nodes as wireframes and occupied children nodes as red

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to draw on.
        """
        if self.children[0] is None:
            if self.status == 'occupied':
                self.draw_cube(ax, facecolor='red', edgecolor='black', alpha=1.0)
        else:
            # Draw container node wireframe first
            self.draw_cube(
                ax, 
                facecolor='white', 
                edgecolor='gray', 
                alpha=0.1)
            for child in self.children:
                child.draw(ax)

# Example usage
if __name__ == "__main__":
    root = OctreeNode(position=(0,0,0), radius=1)

    # Insert some obstacles
    obstacles = []
    for i in range(10):
        obstacle = (random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1))
        obstacles.append(obstacle)

    for obs in obstacles:
        root.insert_obstacle(obs)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    root.draw(ax)
    limit = root.radius * 1.1  # slightly larger than root radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    plt.show()
