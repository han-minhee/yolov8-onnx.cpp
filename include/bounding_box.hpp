#ifndef BOUNDINGBOX_HPP
#define BOUNDINGBOX_HPP

#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <cstdint>
#include <string>

class BoundingBox
{
public:
    float x1, y1, x2, y2;
    size_t class_id;
    float probability;

    BoundingBox(float x1, float y1, float x2, float y2, size_t class_id, float probability)
        : x1(x1), y1(y1), x2(x2), y2(y2), class_id(class_id), probability(probability) {}

    float intersection(const BoundingBox &other) const;
    float union_area(const BoundingBox &other) const;
    float iou(const BoundingBox &other) const;

    bool operator<(const BoundingBox &other) const
    {
        return probability > other.probability;
    }

    static std::vector<BoundingBox> nms(const std::vector<BoundingBox> &boxes, float iou_threshold);
    std::string toString() const;
    std::string toYOLOString(int image_width, int image_height) const;

private:
};

#endif // BOUNDINGBOX_HPP
