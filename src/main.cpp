#include <unordered_map>
#include <vector>

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
    SessionConfig config;
#ifdef USE_HIP
    Device *device = new HipDevice(0);
    config.device = device;
#endif

    Session session("../sample/yolov8n.onnx", config);
    Image image("../sample/people1.jpg");
    image.resize_and_pad(640, 640);
    std::unordered_map<std::string, Tensor> inputs;
    inputs["images"] = image.to_tensor();

    std::cout << "Input Tensor" << std::endl;
    std::cout << inputs["images"].toString() << std::endl;

    std::unordered_map<std::string, Tensor> outputs = session.run(inputs);
    std::cout << "Output Tensor" << std::endl;
    std::cout << outputs["output0"].toString() << std::endl;
    std::vector<BoundingBox> output_boxes = extract_bounding_boxes(outputs["output0"]);
    std::vector<BoundingBox> bboxes = nms(output_boxes, 0.45);

    for (const auto &bbox : bboxes)
    {
        std::cout << bbox.toString() << std::endl;

        draw_box(image, bbox, get_color_for_class_id(bbox.class_id));
    }

    image.write("../sample/people1_output.jpg");

    save_to_yolo_txt("../sample/people1_output.txt", bboxes, image.getWidth(), image.getHeight());
}
