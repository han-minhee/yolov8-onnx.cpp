#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <iostream>
#include <string>
#include <vector>
#include "tensor/tensor.hpp"

class Image
{
public:
    Image(const std::string &filepath);
    ~Image();

    bool load(const std::string &filepath);
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;
    unsigned char *getData() const;
    void displayInfo() const;
    bool resize(int newWidth, int newHeight);
    bool resize_and_pad(int newWidth, int newHeight, std::vector<unsigned char> color = {112, 112, 112});
    bool write(const std::string &filepath) const;
    Tensor to_tensor();
    bool drawRectangle(int x0, int y0, int x1, int y1, std::vector<unsigned char> color);

private:
    int width;
    int height;
    int channels;
    unsigned char *data;
};

#endif
