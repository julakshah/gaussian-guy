import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
import random
import math

# ==========================================
# CONFIGURATION
# ==========================================
WORLD_SIZE = 20.0          # Total width of the world
MIN_VOXEL_SIZE = 0.5       # Resolution of the map
FOV_DEG = 60.0             # Field of View
SENSOR_RES = 15            # Rays per dimension (Low res for Python speed)
MAX_RANGE = 10.0           # Max sensor range
STANDOFF_DIST = 1.5        # How far to stay from the wall (increased for viz)
START_POS = (-8, -8, -5)   # Robot Start

# ==========================================
# 1. OCTREE (Updated with Raycasting)
# ==========================================
class OctreeNode:
    def __init__(self, position, r, parent=None, min_size=MIN_VOXEL_SIZE):
        self.position = position
        self.r = r
        self.parent = parent
        self.children = [None] * 8
        self.status = 'unknown'
        
        # Calculate max depth once to determine leaf size
        # Leaf size is roughly r / 2^depth
        self.min_size = min_size
        self.leaf_size = self._calculate_leaf_size()

    def _calculate_leaf_size(self):
        # Recursively finding depth is slow, so we approximate/hardcode based on r
        # For the root, we want to know how deep we CAN go
        return self.min_size

    def splitting(self):
        if self.r <= self.min_size:
            return
        
        new_r = self.r / 2
        offsets = [
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
        ]
        
        for i, (ox, oy, oz) in enumerate(offsets):
            new_pos = (
                self.position[0] + ox * new_r,
                self.position[1] + oy * new_r,
                self.position[2] + oz * new_r
            )
            self.children[i] = OctreeNode(new_pos, new_r, self, self.min_size)
        
        # If we were unknown, children inherit unknown. 
        # If we were empty, children inherit empty.
        # Occupied usually means we need to refine, so children start unknown? 
        # Simplification: Inherit status
        for child in self.children:
            child.status = self.status

    def find_leaf(self, position):
        if (abs(position[0] - self.position[0]) > self.r or
            abs(position[1] - self.position[1]) > self.r or
            abs(position[2] - self.position[2]) > self.r):
            return None

        if self.children[0] is None:
            return self

        # Determine child index
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        dz = position[2] - self.position[2]
        
        idx = 0
        if dx > 0: idx += 4
        if dy > 0: idx += 2
        if dz > 0: idx += 1
        
        return self.children[idx].find_leaf(position)

    def insert_obstacle(self, pos):
        leaf = self.find_leaf(pos)
        if leaf:
            # Split if too big
            while leaf.r > leaf.min_size:
                leaf.splitting()
                leaf = leaf.find_leaf(pos)
            leaf.status = 'occupied'
            return leaf
        return None

    def insert_free(self, pos):
        """Mark a specific point as free (helper for raycast)"""
        leaf = self.find_leaf(pos)
        if leaf:
            while leaf.r > leaf.min_size:
                leaf.splitting()
                leaf = leaf.find_leaf(pos)
            # Don't overwrite obstacles with free space (unless dynamic)
            if leaf.status != 'occupied':
                leaf.status = 'empty'

    def raycast(self, origin, target):
        """
        Carves FREE space from origin to target.
        """
        diff = np.array(target) - np.array(origin)
        dist = np.linalg.norm(diff)
        
        if dist == 0: return

        direction = diff / dist
        step_size = self.min_size / 2.0 # Nyquist sampling
        
        current_dist = 0.0
        current_pos = np.array(origin, dtype=float)
        
        # Stop 1 step before target to avoid erasing the wall
        while current_dist < (dist - step_size):
            self.insert_free(tuple(current_pos))
            current_pos += direction * step_size
            current_dist += step_size

    def get_neighbors(self, root):
        # Simplified neighbor finding (just checks 6 directions)
        neighbors = []
        dirs = [
            (self.r*2.1, 0, 0), (-self.r*2.1, 0, 0),
            (0, self.r*2.1, 0), (0, -self.r*2.1, 0),
            (0, 0, self.r*2.1), (0, 0, -self.r*2.1)
        ]
        for dx, dy, dz in dirs:
            n_pos = (self.position[0]+dx, self.position[1]+dy, self.position[2]+dz)
            node = root.find_leaf(n_pos)
            if node and node != self:
                neighbors.append(node)
        return neighbors

    def draw(self, ax):
        if self.children[0] is not None:
            for child in self.children:
                child.draw(ax)
        elif self.status == 'occupied':
            self.draw_cube(ax, 'red', 0.6)
        # elif self.status == 'unknown':
        #     self.draw_cube(ax, 'gray', 0.05) # Draw unknown as faint fog?

    def draw_cube(self, ax, color, alpha):
        x, y, z = self.position
        r = self.r
        # Create a cube
        corners = np.array([
            [x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],
            [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r]
        ])
        faces = [
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[7], corners[6], corners[2], corners[3]],
            [corners[0], corners[4], corners[7], corners[3]],
            [corners[1], corners[5], corners[6], corners[2]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[2], corners[3]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.1, edgecolors='k', alpha=alpha))

# ==========================================
# 2. PATH PLANNING (A* + Standoff)
# ==========================================
def calculate_h(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def find_unknown_leaves(node, leaves):
    if node.children[0] is None:
        if node.status == 'unknown':
            leaves.append(node)
    else:
        for child in node.children:
            find_unknown_leaves(child, leaves)

def find_path(start, goal, root):
    start_node = root.find_leaf(start)
    goal_node = root.find_leaf(goal)
    
    if not start_node or not goal_node: return []
    
    open_set = []
    heapq.heappush(open_set, (0, id(start_node), start_node)) # id() as tiebreaker
    came_from = {}
    g_score = {start_node: 0}
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current.position)
                current = came_from[current]
            path.append(start)
            return path[::-1] # Reverse
        
        for neighbor in current.get_neighbors(root):
            if neighbor.status == 'occupied': continue
            
            tentative_g = g_score[current] + calculate_h(current.position, neighbor.position)
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + calculate_h(neighbor.position, goal)
                heapq.heappush(open_set, (f, id(neighbor), neighbor))
                
    return []

def get_next_view_pose(root, current_pos):
    """
    IMPLEMENTATION OF: closest_unknown + go_to_goal
    """
    leaves = []
    find_unknown_leaves(root, leaves)
    
    if not leaves: return None # Done!
    
    # 1. Find Closest Unknown
    # Filter leaves that are unreachable or too far? No, purely greedy for now.
    closest = min(leaves, key=lambda l: calculate_h(l.position, current_pos))
    target_voxel = np.array(closest.position)
    
    # 2. Calculate Standoff Vector
    curr_arr = np.array(current_pos)
    direction = curr_arr - target_voxel
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6: 
        norm_dir = np.array([0,0,1])
    else:
        norm_dir = direction / dist
        
    # 3. Apply Offset (30cm - 1.5m depending on scale)
    goal_arr = target_voxel + (norm_dir * STANDOFF_DIST)
    
    return tuple(goal_arr) # RETURN TUPLE FIX

# ==========================================
# 3. SIMULATION SENSOR (Raycasting Logic)
# ==========================================
def simulate_lidar(robot_pos, true_obstacles, octree):
    """
    Simulates a sensor scan.
    1. Casts rays in a cone.
    2. Intersects with 'True' world (spheres).
    3. Updates Octree (Carve Free + Insert Occupied).
    """
    hit_points = []
    
    # Generate Rays (Simple cone)
    # In real life, use camera intrinsics. Here, simple spherical coords.
    for theta in np.linspace(0, 2*np.pi, SENSOR_RES):
        for phi in np.linspace(0, np.pi, SENSOR_RES): # Full sphere for simplicity
            
            # Direction vector
            dx = math.sin(phi) * math.cos(theta)
            dy = math.sin(phi) * math.sin(theta)
            dz = math.cos(phi)
            direction = np.array([dx, dy, dz])
            
            # Ray-Sphere Intersection with all obstacles
            closest_dist = MAX_RANGE
            hit_obj = False
            
            for center, radius in true_obstacles:
                # Geometric Ray-Sphere Intersection
                # |O + tD - C|^2 = r^2
                oc = np.array(robot_pos) - np.array(center)
                b = 2.0 * np.dot(oc, direction)
                c = np.dot(oc, oc) - radius*radius
                discriminant = b*b - 4*c
                
                if discriminant > 0:
                    t = (-b - math.sqrt(discriminant)) / 2.0
                    if 0 < t < closest_dist:
                        closest_dist = t
                        hit_obj = True
            
            # Process the ray result
            end_point = np.array(robot_pos) + direction * closest_dist
            
            # A. Carve Free Space (The Raycast Update)
            octree.raycast(robot_pos, end_point)
            
            # B. Insert Obstacle (if we hit one)
            if hit_obj:
                octree.insert_obstacle(tuple(end_point))
                hit_points.append(tuple(end_point))
                
    return hit_points

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # Setup Ground Truth (A sphere and a wall)
    true_obstacles = [
        ((0, 0, 0), 2.0),       # Sphere at center
        ((5, 5, 0), 1.5),       # Another blob
        ((-5, 2, 2), 1.0)
    ]
    
    # Initialize Octree
    octree = OctreeNode((0,0,0), WORLD_SIZE)
    current_pos = START_POS
    
    # Visualization
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Loop
    for step in range(20):
        print(f"--- Step {step} ---")
        print(f"Robot Pos: {current_pos}")
        
        # 1. SENSE & UPDATE MAP
        hits = simulate_lidar(current_pos, true_obstacles, octree)
        print(f"Sensor saw {len(hits)} points")
        
        # 2. PLAN (Next Best View)
        goal_pos = get_next_view_pose(octree, current_pos)
        
        if goal_pos is None:
            print("Exploration Complete!")
            break
            
        print(f"Next Goal: {goal_pos}")
        
        # 3. PATHFINDING
        path = find_path(current_pos, goal_pos, octree)
        
        # 4. VISUALIZE
        ax.clear()
        octree.draw(ax) # Draws occupied voxels
        
        # Draw Robot
        ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], c='g', s=100, label='Robot')
        
        # Draw Path
        if path:
            px, py, pz = zip(*path)
            ax.plot(px, py, pz, c='b', linewidth=2, label='Path')
            
        # Draw Sensor Hits (Point Cloud)
        if hits:
            hx, hy, hz = zip(*hits)
            ax.scatter(hx, hy, hz, c='k', s=5, alpha=0.5, label='Scan')

        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)
        ax.set_title(f"Step {step}: Moving to {np.round(goal_pos, 1)}")
        plt.draw()
        plt.pause(0.1)
        
        # 5. EXECUTE (Teleport for simulation)
        # In real code, you would iterate through 'path'
        if path:
            # Move to the first waypoint (or destination for speed)
            current_pos = path[1] if len(path) > 1 else path[0] 
        else:
            print("No path found! (Robot might be stuck)")
            break

    plt.ioff()
    plt.show()