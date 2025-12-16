// Custom octree header
#include <vector>
#include <string>

struct Point3D {
    double x, y, z;
};

class OctreeNode {
public:
    OctreeNode(Point3D position, double r, double min_r = 1.0);

    bool insert_obstacle(Point3D obstacle_pos);
    void raycast(Point3D start, Point3D end);

    std::string get_status();
    std::vector<OctreeNode*> get_children();

    Point3D position;
    double r;
    double min_r;
    std::string status;

    OctreeNode* parent;
    std::vector<OctreeNode*> children;
private:
    void splitting();
    void prune();
    void update_parents();
    OctreeNode* find_leaf(Point3D position);
};