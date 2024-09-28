#include <unordered_map>
#include <vector>
#include <iostream>
#include <chrono> // Include the necessary header for timing

#include "session/session.hpp"
#include "tensor/tensor.hpp"
#include "bounding_box.hpp"
#include "yolo_utils.hpp"
#include "image_utils.hpp"

#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"
#include "device/device.hpp"

#ifdef USE_HIP
#include "device/device_hip.hpp"
#include <hip/hip_runtime.h>
#endif // USE_HIP

#include "parser/npy_parser.hpp"

int main()
{
    // Configuration for the session
    SessionConfig config;
#ifdef USE_HIP
    Device *device = new HipDevice(0);
    config.device = device;
#endif

    // Initialize the session with the given ONNX model
    Session session("../sample/yolov8n.onnx", config);

    // Load the image and preprocess it
    Image image("../sample/people1.jpg");
    image.resize_and_pad(640, 640);

    // Prepare the input tensor
    std::unordered_map<std::string, Tensor> inputs;
    inputs["images"] = image.to_tensor();

#ifdef USE_HIP
    // Move the input tensor to the device
    inputs["images"].to(device);
#endif

    std::cout << "Input Tensor" << std::endl;
    std::cout << inputs["images"].toString() << std::endl;

    // Start timing before running the model
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the model with the prepared inputs
    std::unordered_map<std::string, Tensor> outputs = session.run(inputs);

    // Get the time after running the model
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "Model inference time: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Output Tensor" << std::endl;

#ifdef USE_HIP
    outputs["output0"].to(new CpuDevice());
#endif

    std::cout << outputs["output0"].toString() << std::endl;

    // Extract bounding boxes from the output tensor
    std::vector<BoundingBox> output_boxes = extract_bounding_boxes(outputs["output0"]);
    std::vector<BoundingBox> bboxes = nms(output_boxes, 0.45);

    // Draw bounding boxes and save the output image
    for (const auto &bbox : bboxes)
    {
        std::cout << bbox.toString() << std::endl;
        draw_box(image, bbox, get_color_for_class_id(bbox.class_id));
    }

    image.write("../sample/people1_output.jpg");

    // Save the results in YOLO format
    save_to_yolo_txt("../sample/people1_output.txt", bboxes, image.getWidth(), image.getHeight());
}
