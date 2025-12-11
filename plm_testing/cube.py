"""
This script implements a basic octtree from scratch as a means of learning how an octree works.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Simple Point and Cube
class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

class Cube:
    def __init__(self, center, half):
        self.center = center
        self.half = half

# Octree Node
class Octree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.children = None

    def contains(self, p):
        b = self.boundary
        return (abs(p.x - b.center.x) <= b.half and
                abs(p.y - b.center.y) <= b.half and
                abs(p.z - b.center.z) <= b.half)

    def subdivide(self):
        cx, cy, cz = self.boundary.center.x, self.boundary.center.y, self.boundary.center.z
        h = self.boundary.half / 2
        self.children = []
        for dx in (-h, h):
            for dy in (-h, h):
                for dz in (-h, h):
                    center = Point(cx+dx, cy+dy, cz+dz)
                    self.children.append(Octree(Cube(center, h), self.capacity))

    def insert(self, p):
        if not self.contains(p):
            return False
        if len(self.points) < self.capacity and self.children is None:
            self.points.append(p)
            return True
        if self.children is None:
            self.subdivide()
        for child in self.children:
            if child.insert(p):
                return True
        return False

# Build a tree
root = Octree(Cube(Point(0,0,0), 10))
points = [Point(random.uniform(-10,10),
                random.uniform(-10,10),
                random.uniform(-10,10)) for _ in range(50)]
for p in points:
    root.insert(p)

# Plot points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([p.x for p in points], [p.y for p in points], [p.z for p in points])

# Draw root cube edges
def draw_cube(ax, cube):
    c, h = cube.center, cube.half
    # all 8 corners
    corners = [(c.x+dx*h, c.y+dy*h, c.z+dz*h)
               for dx in (-1,1) for dy in (-1,1) for dz in (-1,1)]
    # connect edges
    for i in range(8):
        for j in range(i+1, 8):
            xi, yi, zi = corners[i]
            xj, yj, zj = corners[j]
            if sum([xi!=xj, yi!=yj, zi!=zj]) == 1:  # share 2 coords
                ax.plot([xi,xj],[yi,yj],[zi,zj], color='gray')

draw_cube(ax, root.boundary)

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.title("Octree Root Cube with Points")
plt.show()
