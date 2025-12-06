"""Mess around with math and code to figure out Octrees"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

class OctreeNode:
    """
    Octree

    status: 'unknown', 'empty', 'occupied'
    """
    def __init__(self, position, r=20, parent=None):
        self.position = position    # (x, y, z)
        self.r = r  # radius
        self.status = 'unknown'
        self.parent = parent        # Parent node for path reconstruction
        self.children = [None] * 8  # Child nodes

    def splitting(self):
        """Split node into smaller nodes"""
        # Check if already split or minimum size
        if (self.children[0] is not None) or (self.r <= 1):
            return
        
        # Values for child nodes
        new_r = self.r / 2

        child_positions = [] # positives are right, up, front
        for xi in (-1, 1):
            for yi in (-1, 1):
                for zi in (-1, 1):
                    child_positions.append((xi, yi, zi))


        # Create child nodes
        for i, pos in enumerate(child_positions):
            self.children[i] = OctreeNode(
                position= (
                    self.position[0] + pos[0] * new_r,
                    self.position[1] + pos[1] * new_r,
                    self.position[2] + pos[2] * new_r,
                ),
                r=new_r,
                parent=self
            )
        self.status = 'internal'
    
    def get_status(self):
        # If children, internal node so status is irrelevant.
        if self.children[0] is not None:
            return 'internal' 
        
        # If no children, return actual data
        return self.status
    
    def prune(self):
        """Removes unecessary child nodes to save memory"""
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

    def insert_obstacle(self, obstacle_pos):
        """Updates octree to contain the given obstacle"""
        current_node = self

        # Check if already occupied
        if self.get_status() == 'occupied':
            return

        # Check if obstacle in node bounds
        if (abs(obstacle_pos[0] - self.position[0]) > self.r or
            abs(obstacle_pos[1] - self.position[1]) > self.r or
            abs(obstacle_pos[2] - self.position[2]) > self.r):
            return False 
        
        # If at minimum size, mark as occupied
        while current_node.r > 1:

            if current_node.children[0] is None:
                current_node.splitting()
            
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
        return True
    
    def find_leaf(self, position):
        """
        Finds the leaf node containing the given position.
        Returns None if out of bounds.
        """
        current_node = self
        
        # Check if in bounds
        if (abs(position[0] - current_node.position[0]) > current_node.r or
            abs(position[1] - current_node.position[1]) > current_node.r or
            abs(position[2] - current_node.position[2]) > current_node.r):
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

    def update_parents(self):
        """Update parent nodes to prune unnecessary children"""
        current_node = self
        while current_node.parent is not None:
            current_node.parent.prune()
            current_node = current_node.parent


    def draw_cube(self, ax, facecolor='red', edgecolor='black', alpha=1.0): ####
        """Draws a cube at this node."""
        x, y, z = self.position
        r = self.r
        corners = [
            (x-r, y-r, z-r), (x-r, y-r, z+r), (x-r, y+r, z-r), (x-r, y+r, z+r),
            (x+r, y-r, z-r), (x+r, y-r, z+r), (x+r, y+r, z-r), (x+r, y+r, z+r),
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
        """Recursively draw the tree:
        - Occupied nodes: solid red
        - Internal nodes: transparent wireframe
        """
        if self.children[0] is None:
            if self.status == 'occupied':
                self.draw_cube(ax, facecolor='red', edgecolor='black', alpha=1.0)
        else:
            # Draw internal node wireframe first
            self.draw_cube(ax, facecolor='white', edgecolor='gray', alpha=0.1)
            for child in self.children:
                child.draw(ax)

# Example usage
if __name__ == "__main__":
    root = OctreeNode(position=(0,0,0), r=20)

    # Insert some obstacles
    obstacles = []
    for i in range(10):
        obstacle = (random.uniform(-20,20), random.uniform(-20,20), random.uniform(-20,20))
        obstacles.append(obstacle)

    for obs in obstacles:
        root.insert_obstacle(obs)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    root.draw(ax)
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)
    ax.set_box_aspect([1,1,1])
    plt.show()
