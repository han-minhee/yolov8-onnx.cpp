#include <gtest/gtest.h>

#include "parser/npy_parser.hpp"
#include "tensor/tensor.hpp"
#include "operator/operators.hpp"

void PrintTo(OperatorExecuteResult result, std::ostream *os)
{
    *os << OperatorUtils::OperatorExecuteResultToString(result);
}

void run_and_check_operator(Operator &op,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor *> outputs,
                            const std::vector<Tensor> &expected,
                            std::unordered_map<std::string, Node::AttributeValue> attributes = {},
                            OperatorExecuteResult expected_execute_result = OperatorExecuteResult::SUCCESS)
{
    OperatorExecuteResult result_code = op.execute(inputs, outputs, attributes);

    ASSERT_EQ(result_code, expected_execute_result);

    if (result_code != OperatorExecuteResult::SUCCESS)
    {
        return;
    }

    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        std::cout << "Output tensor: " << i << std::endl;
        ASSERT_EQ(outputs[i]->getDims(), expected[i].getDims());
        ASSERT_EQ(outputs[i]->getDataType(), expected[i].getDataType());

        switch (outputs[i]->getDataType())
        {
        case TensorDataType::FLOAT32:
        {
            const float *output_data = outputs[i]->data<float>();
            const float *expected_data = expected[i].data<float>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-4);
            }
            break;
        }
        case TensorDataType::INT32:
        {
            const int32_t *output_data = outputs[i]->data<int32_t>();
            const int32_t *expected_data = expected[i].data<int32_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }
        case TensorDataType::INT64:
        {
            const int64_t *output_data = outputs[i]->data<int64_t>();
            const int64_t *expected_data = expected[i].data<int64_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type.");
        }
    }
}

TEST(OperatorTest, AddOperatorBasic)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    Tensor output;
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {7.0, 7.0, 7.0, 7.0, 7.0, 7.0});
    AddOperator add_op;

    run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, AddOperatorBroadcastScalar)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {1}, {10.0});
    Tensor output;
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    AddOperator add_op;

    run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, AddOperatorBroadcastVector)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {1, 2, 4}, std::vector<float>(8, 1.0f));
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output;
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 2, 4}, {2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    AddOperator add_op;

    run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, AddOperatorBroadcastHigherRank)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3, 4, 5}, std::vector<float>(120, 1.0f));
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {1, 4, 5}, std::vector<float>(20, 2.0f));
    Tensor output;
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3, 4, 5}, std::vector<float>(120, 1.0f + 2.0f));
    AddOperator add_op;

    run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, AddOperatorIncompatibleDataTypes)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::INT32, {2, 3}, {6, 5, 4, 3, 2, 1});

    AddOperator add_op;

    Tensor output;
    std::vector<Tensor *> outputs = {&output};

    run_and_check_operator(add_op, {t1, t2}, outputs, {}, {}, OperatorExecuteResult::DATA_TYPE_ERROR);
}

TEST(OperatorTest, AddOperatorIncompatibleShapes)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {4}, {1.0, 2.0, 3.0, 4.0});

    AddOperator add_op;

    Tensor output;
    std::vector<Tensor *> outputs = {&output};

    run_and_check_operator(add_op, {t1, t2}, outputs, {}, {}, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
}

TEST(OperatorTest, SubOperatorBasic)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0});

    Tensor output;
    SubOperator sub_op;

    run_and_check_operator(sub_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, SubOperatorBroadcastingBasic)
{

    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {1}, {5.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {5.0f, 15.0f, 25.0f, 35.0f, 45.0f, 55.0f});

    Tensor output;
    SubOperator sub_op;
    run_and_check_operator(sub_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, SubOperatorIncompatibleBroadcasting)
{

    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor output;
    SubOperator sub_op;

    run_and_check_operator(sub_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
}

TEST(OperatorTest, MulOperatorBasic)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 10.0, 12.0, 12.0, 10.0, 6.0});

    Tensor output;
    MulOperator mul_op;

    run_and_check_operator(mul_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, MulOperatorBroadcas)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3, 4}, std::vector<float>(24, 2.0f));
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {1, 3, 1}, {1.0f, 2.0f, 3.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3, 4}, {2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 6.0f, 6.0f, 6.0f, 6.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 4.0f, 4.0f, 4.0f, 6.0f, 6.0f, 6.0f, 6.0f});

    Tensor output;
    MulOperator mul_op;

    run_and_check_operator(mul_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, DivOperatorBasic)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0 / 6.0, 2.0 / 5.0, 3.0 / 4.0, 4.0 / 3.0, 5.0 / 2.0, 6.0 / 1.0});

    Tensor output;
    DivOperator div_op;

    run_and_check_operator(div_op, {t1, t2}, {&output}, {expected});
}

TEST(OperatorTest, DivOperatorBroadcastScalar)
{

    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {1}, {10.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor output;
    DivOperator div_op;

    run_and_check_operator(div_op, {t1, t2}, {&output}, {expected});
}

// Division by zero check is currently disabled
// TEST(OperatorTest, DivOperatorZeroDivision)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0, 0.0, 4.0, 3.0, 2.0, 1.0});

//     Tensor output;
//     DivOperator div_op;

//     run_and_check_operator(div_op, {t1, t2}, {&output}, {}, {}, OperatorExecuteResult::DIVIDE_BY_ZERO_ERROR);
// }

TEST(OperatorTest, ConstantOperator0)
{
    ConstantOperator const_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    Tensor output;

    Tensor value_tensor = NpyParser::load("../tests/data/npy/model.22.dfl.conv.weight.npy");
    attributes["value"] = value_tensor;

    run_and_check_operator(const_op, {}, {&output}, {value_tensor}, attributes);
}

TEST(OperatorTest, ConvOperator0)
{
    Tensor X = NpyParser::load("../tests/data/npy/images.npy");
    Tensor W = NpyParser::load("../tests/data/npy/model.0.conv.weight.npy");
    Tensor B = NpyParser::load("../tests/data/npy/model.0.conv.bias.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.0.conv.Conv_output_0.npy");

    Tensor Y;
    ConvOperator conv_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["group"] = 1;
    attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    attributes["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    attributes["strides"] = std::vector<int64_t>{2, 2};

    run_and_check_operator(conv_op, {X, W, B}, {&Y}, {expectedOutput}, attributes);
}

TEST(OperatorTest, ConvOperator1)
{
    Tensor X = NpyParser::load("../tests/data/npy/model.22.dfl.Softmax_output_0.npy");
    Tensor W = NpyParser::load("../tests/data/npy/model.22.dfl.conv.weight.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.dfl.conv.Conv_output_0.npy");

    Tensor Y;

    ConvOperator conv_op;

    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["group"] = 1;
    attributes["kernel_shape"] = std::vector<int64_t>{1, 1};
    attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    attributes["strides"] = std::vector<int64_t>{1, 1};

    run_and_check_operator(conv_op, {X, W}, {&Y}, {expectedOutput}, attributes);
}

TEST(OperatorTest, SoftmaxOperator0)
{

    Tensor input = NpyParser::load("../tests/data/npy/model.22.dfl.Transpose_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.dfl.Softmax_output_0.npy");

    Tensor output;
    SoftmaxOperator softmax_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    run_and_check_operator(softmax_op, {input}, {&output}, {expectedOutput}, attributes);
}

TEST(OperatorTest, SplitOperator0)
{
    Tensor input = NpyParser::load("../tests/data/npy/model.22.Concat_3_output_0.npy");
    Tensor split = NpyParser::load("../tests/data/npy/onnx..Split_388.npy");
    Tensor expectedOutput0 = NpyParser::load("../tests/data/npy/model.22.Split_output_0.npy");
    Tensor expectedOutput1 = NpyParser::load("../tests/data/npy/model.22.Split_output_1.npy");

    Tensor output0;
    Tensor output1;

    SplitOperator split_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    std::vector<Tensor> inputs = {input, split};
    std::vector<Tensor *> outputs = {&output0, &output1};

    run_and_check_operator(split_op, inputs, outputs, {expectedOutput0, expectedOutput1}, attributes);
}

TEST(OperatorTest, MaxPoolOperator0)
{
    //     ceil_mode
    // 0
    // dilations
    // 1, 1
    // kernel_shape
    // 5, 5
    // pads
    // 2, 2, 2, 2
    // strides
    // 1, 1
    // X
    // name: /model.9/cv1/act/Mul_output_0
    // Y
    // name: /model.9/m/MaxPool_output_0

    Tensor X = NpyParser::load("../tests/data/npy/model.9.cv1.act.Mul_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.9.m.MaxPool_output_0.npy");

    MaxPoolOperator max_pool_op;

    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["ceil_mode"] = 0;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["kernel_shape"] = std::vector<int64_t>{5, 5};
    attributes["pads"] = std::vector<int64_t>{2, 2, 2, 2};
    attributes["strides"] = std::vector<int64_t>{1, 1};

    // get the expected output shape and data type
    std::vector<size_t> expected_output_shape = max_pool_op.inferOutputShapes({X}, attributes)[0];
    TensorDataType expected_output_data_type = max_pool_op.inferOutputDataTypes({X}, attributes)[0];
    Tensor Y = Tensor(expected_output_data_type, expected_output_shape);

    run_and_check_operator(max_pool_op, {X}, {&Y}, {expectedOutput}, attributes);
}

// Executing node: /model.22/Split
// Getting input tensor: /model.22/Concat_3_output_0
// Getting input tensor: onnx::Split_388
// Found 2 input tensors:
// Tensor: dtype=FLOAT32, dims=[1, 144, 8400], data=[7.7661, 3.71506, 1.8796, 1.22822, 0.187344, 0.124847, ...]
// Tensor: dtype=INT64, dims=[2], data=[64, 80]
// Infering output shapes
// Inferring output data type
// Num outputs: 2
// Inferred output shapes and data types
// Output shapes:
// 1 64 8400
// 1 80 8400
// Allocating output tensors
// Dtypes:
// FLOAT32
// FLOAT32
// Output shapes:
// Allocating output tensor: /model.22/Split_output_0
// For dtype: FLOAT32
// Getting or allocating intermediate tensor: /model.22/Split_output_0
// Allocating intermediate tensor
// Dims:
// 1 64 8400 Intermediate tensor allocated
// Allocated output tensor: /model.22/Split_output_0
// Allocating output tensor: /model.22/Split_output_1
// For dtype: FLOAT32
// Getting or allocating intermediate tensor: /model.22/Split_output_1
// Allocating intermediate tensor
// Dims:
// 1 80 8400 Intermediate tensor allocated
// Allocated output tensor: /model.22/Split_output_1
// Executing operator: Split
// Output tensor: /model.22/Split_output_0
// Tensor: dtype=FLOAT32, dims=[1, 64, 8400], data=[7.7661, 3.71506, 1.8796, 1.22822, 0.187344, 0.124847, ...]
// Output tensor: /model.22/Split_output_1
// Tensor: dtype=FLOAT32, dims=[1, 80, 8400], data=[-15.7214, -16.153, -15.7761, -15.6962, -15.5299, -15.6678, ...]

TEST(OperatorTest, SplitOperator1)
{
    Tensor input = NpyParser::load("../tests/data/npy/model.22.Concat_3_output_0.npy");
    Tensor split = NpyParser::load("../tests/data/npy/onnx..Split_388.npy");
    Tensor expectedOutput0 = NpyParser::load("../tests/data/npy/model.22.Split_output_0.npy");
    Tensor expectedOutput1 = NpyParser::load("../tests/data/npy/model.22.Split_output_1.npy");

    Tensor output0;
    Tensor output1;

    SplitOperator split_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = static_cast<int64_t>(1);

    std::vector<Tensor> inputs = {input, split};
    std::vector<Tensor *> outputs = {&output0, &output1};

    run_and_check_operator(split_op, inputs, outputs, {expectedOutput0, expectedOutput1}, attributes);
}

// Executing node: /model.0/act/Sigmoid
// Getting input tensor: /model.0/conv/Conv_output_0
// Found 1 input tensors:
// Tensor: dtype=FLOAT32, dims=[1, 16, 320, 320], data=[36.4291, 1.9317, 1.9317, 1.9317, 1.9317, 1.9317, ...
// Infering output shapes
// Inferring output data type
// Inferred output shapes and data types
// Output shapes:
// 1 16 320 320
// Allocating output tensors
// Dtypes:
// FLOAT32
// Output shapes:
// Allocating output tensor: /model.0/act/Sigmoid_output_0
// For dtype: FLOAT32
// Getting or allocating intermediate tensor: /model.0/act/Sigmoid_output_0
// Allocating intermediate tensor
// Dims:
// 1 16 320 320 Calculating number of elements
// Tensor constructor called
// Allocating data
// Data allocated
// Intermediate tensor allocated
// Allocated output tensor: /model.0/act/Sigmoid_output_0
// Executing operator: Sigmoid
// Calculating number of elements
// Output tensor: /model.0/act/Sigmoid_output_0
// Tensor: dtype=FLOAT32, dims=[1, 16, 320, 320], data=[1, 0.873438, 0.873438, 0.873438, 0.873438, 0.873438, ...

TEST(OperatorTest, ConvOperator2)
{
    //     dilations
    // 1, 1
    // group
    // 1
    // kernel_shape
    // 3, 3
    // pads
    // 1, 1, 1, 1
    // strides
    // 2, 2
    // X
    // name: images
    // W
    // name: model.0.conv.weight
    // B
    // name: model.0.conv.bias
    // Y
    // name: /model.0/conv/Conv_output_0

    Tensor X = NpyParser::load("../tests/data/npy/images.npy");
    Tensor W = NpyParser::load("../tests/data/npy/model.0.conv.weight.npy");
    Tensor B = NpyParser::load("../tests/data/npy/model.0.conv.bias.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.0.conv.Conv_output_0.npy");

    Tensor Y;
    ConvOperator conv_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["group"] = 1;
    attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    attributes["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    attributes["strides"] = std::vector<int64_t>{2, 2};

    run_and_check_operator(conv_op, {X, W, B}, {&Y}, {expectedOutput}, attributes);
}

TEST(OperatorTest, SigmoidOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.0.conv.Conv_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.0.act.Sigmoid_output_0.npy");

    Tensor output;
    SigmoidOperator sigmoid_op;

    run_and_check_operator(sigmoid_op, {input0}, {&output}, {expectedOutput});
}

TEST(OperatorTest, ShapeOperator0)
{

    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.dfl.Reshape_1_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.Shape_output_0.npy");
    Tensor output;
    ShapeOperator shape_op;

    run_and_check_operator(shape_op, {input0}, {&output}, {expectedOutput});
}

TEST(OperatorTest, GatherOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.Shape_output_0.npy");
    Tensor input1 = NpyParser::load("../tests/data/npy/model.22.Constant_3_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.Gather_output_0.npy");

    Tensor output;
    GatherOperator gather_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    run_and_check_operator(gather_op, {input0, input1}, {&output}, {expectedOutput}, attributes);
}

TEST(OperatorTest, TransposeOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.dfl.Reshape_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.dfl.Transpose_output_0.npy");

    Tensor output;
    TransposeOperator transpose_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["perm"] = std::vector<int64_t>{0, 2, 1, 3};

    run_and_check_operator(transpose_op, {input0}, {&output}, {expectedOutput}, attributes);
}

TEST(OperatorTest, ResizeOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.9.cv2.act.Mul_output_0.npy");
    Tensor input1;
    Tensor input2 = NpyParser::load("../tests/data/npy/model.10.Constant_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.10.Resize_output_0.npy");

    Tensor Y;
    ResizeOperator resize_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["coordinate_transformation_mode"] = std::string("asymmetric");
    attributes["cubic_coeff_a"] = static_cast<float>(-0.75f);
    attributes["mode"] = std::string("nearest");
    attributes["nearest_mode"] = std::string("floor");

    run_and_check_operator(resize_op, {input0, input1, input2}, {&Y}, {expectedOutput}, attributes);
}

TEST(OperatorTest, ReshapeOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.dfl.conv.Conv_output_0.npy");
    Tensor input1 = NpyParser::load("../tests/data/npy/model.22.dfl.Constant_1_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.dfl.Reshape_1_output_0.npy");

    Tensor output;
    ReshapeOperator reshape_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["allowzero"] = 0;

    run_and_check_operator(reshape_op, {input0, input1}, {&output}, {expectedOutput}, attributes);
}

TEST(OperatorTest, SliceOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.dfl.Reshape_1_output_0.npy");
    Tensor input1 = NpyParser::load("../tests/data/npy/model.22.Constant_4_output_0.npy");
    Tensor input2 = NpyParser::load("../tests/data/npy/model.22.Mul_output_0.npy");
    Tensor input3 = NpyParser::load("../tests/data/npy/model.22.Constant_3_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.Slice_output_0.npy");

    Tensor output;
    SliceOperator slice_op;

    run_and_check_operator(slice_op, {input0, input1, input2, input3}, {&output}, {expectedOutput});
}

// name
// /model.10/Constant_output_0
// category
// Constant
// type
// float32
// shape
// 4
// value

// [
//     1,
//     1,
//     2,
//     2
// ]

TEST(OperatorTest, ConstantOperator1)
{
    ConstantOperator const_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    Tensor output;

    Tensor value_tensor = NpyParser::load("../tests/data/npy/model.10.Constant_output_0.npy");
    attributes["value"] = value_tensor;

    run_and_check_operator(const_op, {}, {&output}, {value_tensor}, attributes);
}

TEST(OperatorTest, ConactOperator0)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.Mul_2_output_0.npy");
    Tensor input1 = NpyParser::load("../tests/data/npy/model.22.Sigmoid_output_0.npy");
    Tensor expectedOutput = NpyParser::load("../tests/data/npy/output0.npy");

    ConcatOperator concat_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    // set the output tensor shape and data type
    std::vector<size_t> expected_output_shape = concat_op.inferOutputShapes({input0, input1}, attributes)[0];
    TensorDataType expected_output_data_type = concat_op.inferOutputDataTypes({input0, input1}, attributes)[0];

    Tensor output = Tensor(expected_output_data_type, expected_output_shape);

    run_and_check_operator(concat_op, {input0, input1}, {&output}, {expectedOutput}, attributes);
}

// axis
// 2
// inputs
// name: /model.22/Reshape_output_0
// name: /model.22/Reshape_1_output_0
// name: /model.22/Reshape_2_output_0
// concat_result
// name: /model.22/Concat_3_output_0

TEST(OperatorTEST, ConcatOperator1)
{
    Tensor input0 = NpyParser::load("../tests/data/npy/model.22.Reshape_output_0.npy");
    Tensor input1 = NpyParser::load("../tests/data/npy/model.22.Reshape_1_output_0.npy");
    Tensor input2 = NpyParser::load("../tests/data/npy/model.22.Reshape_2_output_0.npy");

    Tensor expectedOutput = NpyParser::load("../tests/data/npy/model.22.Concat_3_output_0.npy");

    ConcatOperator concat_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 2;

    // set the output tensor shape and data type
    std::vector<size_t> expected_output_shape = concat_op.inferOutputShapes({input0, input1, input2}, attributes)[0];
    TensorDataType expected_output_data_type = concat_op.inferOutputDataTypes({input0, input1, input2}, attributes)[0];

    Tensor output = Tensor(expected_output_data_type, expected_output_shape);

    run_and_check_operator(concat_op, {input0, input1, input2}, {&output}, {expectedOutput}, attributes);
}
