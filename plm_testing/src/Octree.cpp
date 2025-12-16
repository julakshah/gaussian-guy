// Custom octree
# include "Octree.h"
# include <cmath>

OctreeNode::OctreeNode(Point3D position, double r, double min_r): position(position), r(r), min_r(min_r), parent(nullptr) {
    
    // Initialize status
    this->status = "unknown";

    // Children holds 8 null pointers
    this->children.resize(8, nullptr);
}