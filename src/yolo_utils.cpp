#include "yolo_utils.hpp"

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

std::vector<BoundingBox> extract_bounding_boxes(const Tensor &output, float confidence_threshold)
{
    if (output.getDataType() != TensorDataType::FLOAT32)
    {
        throw std::runtime_error("The output tensor must be of type float32");
    }

    const auto &dims = output.getDims();
    if (dims.size() != 3)
    {
        throw std::runtime_error("The output tensor must have 3 dimensions");
    }
    // float32[1,84,8400]

    // check if dims are correct
    if (dims != std::vector<size_t>{1, 84, 8400})
    {
        throw std::runtime_error("The output tensor must have the shape [1, 84, 8400]");
    }

    size_t num_classes = dims[1] - 4; // total 80 classes
    size_t num_predictions = dims[2]; // 8400 detections, of which has x, y, w, h, and 80 class probabilities

    const std::vector<size_t> strides = output.getStrides();
    const float *data = output.data<float>();
    std::vector<BoundingBox> boxes;

    // std::vector<size_t> Tensor::calcStrides(const std::vector<size_t> &dims)
    // {
    //     std::vector<size_t> stride(dims.size(), 1);
    //     for (int i = dims.size() - 2; i >= 0; --i)
    //     {
    //         stride[i] = stride[i + 1] * dims[i + 1];
    //     }
    //     return stride;
    // }
    for (size_t i = 0; i < num_predictions; ++i)
    {
        // Extract x, y, w, h
        float x = data[0 * strides[1] + i];
        float y = data[1 * strides[1] + i];
        float w = data[2 * strides[1] + i];
        float h = data[3 * strides[1] + i];

        // Find the class with the highest probability
        float max_class_prob = 0.0f;
        size_t max_class_id = 0;

        for (size_t j = 0; j < num_classes; ++j)
        {
            float class_prob = data[(4 + j) * strides[1] + i];
            if (class_prob > max_class_prob)
            {
                max_class_prob = class_prob;
                max_class_id = j;
            }
        }
        if (max_class_prob > confidence_threshold)
        {
            BoundingBox bbox(
                x - w / 2.0f,
                y - h / 2.0f,
                x + w / 2.0f,
                y + h / 2.0f,
                max_class_id,
                max_class_prob);
            boxes.push_back(bbox);
        }
    }
    return boxes;
}

std::vector<BoundingBox> nms(const std::vector<BoundingBox> &boxes, float iou_threshold)
{
    std::vector<BoundingBox> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end());

    std::vector<BoundingBox> result;
    while (!sorted_boxes.empty())
    {
        BoundingBox best_box = sorted_boxes.front();
        sorted_boxes.erase(sorted_boxes.begin());
        result.push_back(best_box);

        sorted_boxes.erase(std::remove_if(sorted_boxes.begin(), sorted_boxes.end(),
                                          [&best_box, iou_threshold](const BoundingBox &bbox)
                                          {
                                              return best_box.iou(bbox) >= iou_threshold;
                                          }),
                           sorted_boxes.end());
    }

    return result;
}

std::vector<unsigned char> get_color_for_class_id(int class_id)
{
    float hue = (static_cast<float>(class_id) / 80) * 360.0f;
    auto [r, g, b] = hsv_to_rgb(hue, 1.0f, 1.0f);
    return std::vector<unsigned char>{static_cast<uint8_t>(r * 255.0f),
                                      static_cast<uint8_t>(g * 255.0f),
                                      static_cast<uint8_t>(b * 255.0f),
                                      0xFF};
}

std::tuple<float, float, float> hsv_to_rgb(float h, float s, float v)
{
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2) - 1.0f));
    float m = v - c;

    float r = 0, g = 0, b = 0;
    if (h >= 0 && h < 60)
    {
        r = c;
        g = x;
        b = 0;
    }
    else if (h >= 60 && h < 120)
    {
        r = x;
        g = c;
        b = 0;
    }
    else if (h >= 120 && h < 180)
    {
        r = 0;
        g = c;
        b = x;
    }
    else if (h >= 180 && h < 240)
    {
        r = 0;
        g = x;
        b = c;
    }
    else if (h >= 240 && h < 300)
    {
        r = x;
        g = 0;
        b = c;
    }
    else if (h >= 300 && h < 360)
    {
        r = c;
        g = 0;
        b = x;
    }

    return std::make_tuple(r + m, g + m, b + m);
}

void draw_box(Image &image, const BoundingBox &bbox, std::vector<unsigned char> color)
{
    int x0 = bbox.x1;
    int y0 = bbox.y1;
    int x1 = bbox.x2;
    int y1 = bbox.y2;

    image.drawRectangle(x0, y0, x1, y1, color);
}

void draw_boxes(Image &image, const std::vector<BoundingBox> &boxes, std::vector<unsigned char> color)
{
    for (const auto &bbox : boxes)
    {
        int x0 = bbox.x1 * image.getWidth();
        int y0 = bbox.y1 * image.getHeight();
        int x1 = bbox.x2 * image.getWidth();
        int y1 = bbox.y2 * image.getHeight();

        image.drawRectangle(x0, y0, x1, y1, color);
    }
}

void save_to_yolo_txt(const std::string &filename, const std::vector<BoundingBox> &boxes, int width, int height)
{
    std::ofstream outfile(filename);

    if (!outfile.is_open())
    {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    for (const auto &bbox : boxes)
    {
        std::string yolo_string = bbox.toYOLOString(width, height);
        outfile << yolo_string << '\n';
    }

    outfile.close();
}