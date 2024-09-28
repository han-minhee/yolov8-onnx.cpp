#include "image_utils.hpp"

#include <fstream>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image/stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

Image::Image(const std::string &filepath) : width(0), height(0), channels(0), data(nullptr)
{
    load(filepath);
}

Image::~Image()
{
    if (data)
    {
        stbi_image_free(data);
    }
}

bool Image::load(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "File does not exist: " << filepath << std::endl;
        return false;
    }

    if (data)
    {
        stbi_image_free(data);
        data = nullptr;
    }

    data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if (data == nullptr)
    {
        std::cerr << "Failed to load image: " << filepath << std::endl;
        return false;
    }
    return true;
}

int Image::getWidth() const { return width; }
int Image::getHeight() const { return height; }
int Image::getChannels() const { return channels; }
unsigned char *Image::getData() const { return data; }

void Image::displayInfo() const
{
    if (data)
    {
        std::cout << "Width: " << width << "\n";
        std::cout << "Height: " << height << "\n";
        std::cout << "Channels: " << channels << "\n";
    }
    else
    {
        std::cout << "No image data available.\n";
    }
}

bool Image::resize(int newWidth, int newHeight)
{
    if (!data)
    {
        std::cerr << "No image data available to resize.\n";
        return false;
    }

    unsigned char *resizedData = new unsigned char[newWidth * newHeight * channels];

    stbir_pixel_layout pixel_layout;
    switch (channels)
    {
    case 1:
        pixel_layout = STBIR_1CHANNEL;
        break;
    case 2:
        pixel_layout = STBIR_2CHANNEL;
        break;
    case 3:
        pixel_layout = STBIR_RGB;
        break;
    case 4:
        pixel_layout = STBIR_RGBA;
        break;
    default:
        std::cerr << "Unsupported channel count: " << channels << "\n";
        delete[] resizedData;
        return false;
    }

    if (!stbir_resize_uint8_linear(data, width, height, width * channels, resizedData, newWidth, newHeight, newWidth * channels, pixel_layout))
    {
        std::cerr << "Failed to resize the image.\n";
        delete[] resizedData;
        return false;
    }

    stbi_image_free(data);
    data = resizedData;
    width = newWidth;
    height = newHeight;
    return true;
}

bool Image::resize_and_pad(int newWidth, int newHeight, std::vector<unsigned char> color)
{
    if (!data)
    {
        std::cerr << "No image data available to resize and pad.\n";
        return false;
    }

    float aspectRatioOriginal = static_cast<float>(width) / height;
    float aspectRatioNew = static_cast<float>(newWidth) / newHeight;

    int resizedWidth, resizedHeight;
    if (aspectRatioOriginal > aspectRatioNew)
    {
        resizedWidth = newWidth;
        resizedHeight = static_cast<int>(newWidth / aspectRatioOriginal);
    }
    else
    {
        resizedHeight = newHeight;
        resizedWidth = static_cast<int>(newHeight * aspectRatioOriginal);
    }

    unsigned char *resizedData = new unsigned char[resizedWidth * resizedHeight * channels];

    stbir_pixel_layout pixel_layout;
    switch (channels)
    {
    case 1:
        pixel_layout = STBIR_1CHANNEL;
        break;
    case 2:
        pixel_layout = STBIR_2CHANNEL;
        break;
    case 3:
        pixel_layout = STBIR_RGB;
        break;
    case 4:
        pixel_layout = STBIR_RGBA;
        break;
    default:
        std::cerr << "Unsupported channel count: " << channels << "\n";
        delete[] resizedData;
        return false;
    }

    if (!stbir_resize_uint8_linear(data, width, height, width * channels,
                                   resizedData, resizedWidth, resizedHeight, resizedWidth * channels, pixel_layout))
    {
        std::cerr << "Failed to resize the image.\n";
        delete[] resizedData;
        return false;
    }

    unsigned char *paddedData = new unsigned char[newWidth * newHeight * channels];
    for (int y = 0; y < newHeight; ++y)
    {
        for (int x = 0; x < newWidth; ++x)
        {
            int index = (y * newWidth + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                paddedData[index + c] = color[c % 3];
            }
        }
    }

    int offsetX = (newWidth - resizedWidth) / 2;
    int offsetY = (newHeight - resizedHeight) / 2;

    for (int y = 0; y < resizedHeight; ++y)
    {
        for (int x = 0; x < resizedWidth; ++x)
        {
            int paddedIndex = ((y + offsetY) * newWidth + (x + offsetX)) * channels;
            int resizedIndex = (y * resizedWidth + x) * channels;

            for (int c = 0; c < channels; ++c)
            {
                paddedData[paddedIndex + c] = resizedData[resizedIndex + c];
            }
        }
    }

    delete[] resizedData;
    stbi_image_free(data);
    data = paddedData;
    width = newWidth;
    height = newHeight;
    return true;
}

bool Image::write(const std::string &filepath) const
{
    if (!data)
    {
        std::cerr << "No image data to write.\n";
        return false;
    }

    std::string extension = filepath.substr(filepath.find_last_of(".") + 1);

    bool success = false;
    if (extension == "png")
    {
        success = stbi_write_png(filepath.c_str(), width, height, channels, data, width * channels);
    }
    else if (extension == "jpg" || extension == "jpeg")
    {
        success = stbi_write_jpg(filepath.c_str(), width, height, channels, data, 100);
    }
    else
    {
        std::cerr << "Unsupported file format: " << extension << "\n";
        return false;
    }
    return success;
}

Tensor Image::to_tensor()
{
    if (!data)
    {
        throw std::runtime_error("No image data available to convert to tensor.");
    }

    std::vector<size_t> tensor_dims = {static_cast<size_t>(1), static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)};
    size_t total_elements = tensor_dims[0] * tensor_dims[1] * tensor_dims[2] * tensor_dims[3];

    // float *tensor_data = new float[total_elements];
    std::vector<float> tensor_data(total_elements);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                size_t image_index = (y * width + x) * channels + c;
                size_t tensor_index = c * (height * width) + y * width + x;

                tensor_data[tensor_index] = static_cast<float>(data[image_index]) / 255.0f;
            }
        }
    }

    Tensor tensor(TensorDataType::FLOAT32, tensor_dims, tensor_data);

    return tensor;
}

bool Image::drawRectangle(int x0, int y0, int x1, int y1, std::vector<unsigned char> color)
{
    if (!data)
    {
        std::cerr << "No image data available to draw on.\n";
        return false;
    }

    x0 = std::max(0, std::min(x0, width - 1));
    y0 = std::max(0, std::min(y0, height - 1));
    x1 = std::max(0, std::min(x1, width - 1));
    y1 = std::max(0, std::min(y1, height - 1));

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {

            if (x == x0 || x == x1 || y == y0 || y == y1)
            {
                int index = (y * width + x) * channels;
                if (channels >= 3)
                {
                    data[index] = color[0];
                    data[index + 1] = color[1];
                    data[index + 2] = color[2];
                }
                if (channels == 4)
                {
                    data[index + 3] = 255;
                }
            }
        }
    }

    return true;
}