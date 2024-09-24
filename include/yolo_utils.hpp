#ifndef YOLO_UTILS_HPP
#define YOLO_UTILS_HPP

#include <vector>

#include "bounding_box.hpp"
#include "tensor/tensor.hpp"
#include "image_utils.hpp"

std::vector<BoundingBox> extract_bounding_boxes(const Tensor &output, float confidence_threshold = 0.25);
std::vector<BoundingBox> nms(const std::vector<BoundingBox> &boxes, float iou_threshold);

void draw_box(Image &image, const BoundingBox &box, std::vector<unsigned char> color);
void draw_boxes(Image &image, const std::vector<BoundingBox> &boxes, std::vector<unsigned char> color);
void save_to_yolo_txt(const std::string &filename, const std::vector<BoundingBox> &boxes, int width, int height);

std::vector<unsigned char> get_color_for_class_id(int class_id);
std::tuple<float, float, float> hsv_to_rgb(float h, float s, float v);

#endif // YOLO_UTILS_HPP