#include "bounding_box.hpp"
#include <cmath>

float BoundingBox::intersection(const BoundingBox &other) const
{
    float x_overlap = std::max(0.0f, std::min(x2, other.x2) - std::max(x1, other.x1));
    float y_overlap = std::max(0.0f, std::min(y2, other.y2) - std::max(y1, other.y1));
    return x_overlap * y_overlap;
}

float BoundingBox::union_area(const BoundingBox &other) const
{
    float intersection_area = intersection(other);
    float area_self = (x2 - x1) * (y2 - y1);
    float area_other = (other.x2 - other.x1) * (other.y2 - other.y1);
    return area_self + area_other - intersection_area;
}

float BoundingBox::iou(const BoundingBox &other) const
{
    return intersection(other) / union_area(other);
}

std::string BoundingBox::toString() const
{
    return "BoundingBox(x1=" + std::to_string(x1) + ", y1=" + std::to_string(y1) + ", x2=" + std::to_string(x2) + ", y2=" + std::to_string(y2) + ", class_id=" + std::to_string(class_id) + ", probability=" + std::to_string(probability) + ")";
}

std::string BoundingBox::toYOLOString(int image_width, int image_height) const
{
    // 0 0.5863341 0.39326453 0.30159035 0.37726763
    // class_id x_center y_center width height
    float x_center = (x1 + x2) / 2.0f;
    x_center /= image_width;
    float y_center = (y1 + y2) / 2.0f;
    y_center /= image_height;

    float width = (x2 - x1) / image_width;
    float height = (y2 - y1) / image_height;

    return std::to_string(class_id) + " " + std::to_string(x_center) + " " + std::to_string(y_center) + " " + std::to_string(width) + " " + std::to_string(height);
}
