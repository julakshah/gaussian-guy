"""Mess around with math and code to figure out Octrees"""

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
            
            # child_positions = [] # positives are right, up, front
            # for xi in (-1, 1):
            #     for yi in (-1, 1):
            #         for zi in (-1, 1):
            #             child_positions.append((xi, yi, zi))


            child_index = x_index * 4 + y_index * 2 + z_index
            current_node = current_node.children[child_index]

        current_node.status = "occupied"
        return True

    def update_parents(self):
        """Update parent nodes to prune unnecessary children"""
        current_node = self
        while current_node.parent is not None:
            current_node.parent.prune()
            current_node = current_node.parent