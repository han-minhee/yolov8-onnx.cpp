#include <gtest/gtest.h>
#include "session/session.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "enums.hpp"

#include "parser/npy_parser.hpp"

void PrintTo(TensorCompareResult result, std::ostream *os)
{
    *os << TensorUtils::TensorCompareResultToString(result);
}

TEST(SessionTest, RunCPUSessionWithValidation)
{
    Device *device = new CpuDevice();
    Session session("../tests/data/onnx/yolov8n.onnx", {device, false, {}});
    std::unordered_map<std::string, Tensor> inputs;
    inputs["images"] = NpyParser::load("../tests/data/npy/images.npy");
    std::unordered_map<std::string, Tensor> outputs = session.runWithValidation(inputs);
    ASSERT_FALSE(outputs.empty());
    for (const auto &output_pair : outputs)
    {
        std::string output_name = output_pair.first;
        Tensor output_tensor = output_pair.second;

        std::string sanitized_name = sanitizeFileName(output_name);
        std::string reference_path = "../tests/data/npy/" + sanitized_name + ".npy";

        Tensor expected_tensor = NpyParser::load(reference_path);

        TensorCompareResult compare_result = TensorUtils::areTensorsEqual(output_tensor, expected_tensor);
        ASSERT_EQ(compare_result, TensorCompareResult::EQUAL);
    }
}

#ifdef USE_HIP
TEST(SessionTest, RunHIPSessionWithValidation)
{
    Device *device = new HipDevice(0);
    Session session("../tests/data/onnx/yolov8n.onnx", {device, false, {}});
    std::unordered_map<std::string, Tensor> inputs;
    inputs["images"] = NpyParser::load("../tests/data/npy/images.npy");
    inputs["images"].to(device);
    std::unordered_map<std::string, Tensor> outputs = session.runWithValidation(inputs);
    ASSERT_FALSE(outputs.empty());
    for (const auto &output_pair : outputs)
    {
        std::string output_name = output_pair.first;
        Tensor output_tensor = output_pair.second;

        std::string sanitized_name = sanitizeFileName(output_name);
        std::string reference_path = "../tests/data/npy/" + sanitized_name + ".npy";

        Tensor expected_tensor = NpyParser::load(reference_path);

        TensorCompareResult compare_result = TensorUtils::areTensorsEqual(output_tensor, expected_tensor);
        ASSERT_EQ(compare_result, TensorCompareResult::EQUAL);
    }
}
#endif // USE_HIP