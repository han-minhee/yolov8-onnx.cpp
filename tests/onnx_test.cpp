#include <gtest/gtest.h>

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "parser/onnx_parser.hpp"
#include "session/session.hpp"

TEST(ONNXTest, OnnxSort_Test0)
{
    /// FIXME: currently, there is a method to parse an ONNX file and return a Graph object
    // and another one that makes a session from an ONNX file.
    // parseONNX is currently not used inside the program
    Graph onnxGraph = parseONNX("../tests/data/onnx/yolov8n.onnx");
    onnxGraph.topologicalSort();
    std::vector<Node> sortedNodes = onnxGraph.getNodes();
    // 0. Get the index for the node name  /model.0/conv/Conv and it should come first
    // 1. Get the index for the node name  /model.0/act/Sigmoid and it should come second
    // 2. /model.22/Concat_5 should be the last

    // First, remove all nodes with optype Constant
    for (size_t i = 0; i < sortedNodes.size(); ++i)
    {
        if (sortedNodes[i].getOpType() == "Constant")
        {
            sortedNodes.erase(sortedNodes.begin() + i);
            --i;
        }
    }

    for (size_t i = 0; i < sortedNodes.size(); ++i)
    {
        if (sortedNodes[i].getName() == "/model.0/conv/Conv")
        {
            EXPECT_EQ(i, 0);
        }

        if (sortedNodes[i].getName() == "/model.0/act/Sigmoid")
        {
            EXPECT_EQ(i, 1);
        }

        if (sortedNodes[i].getName() == "/model.22/Concat_5")
        {
            EXPECT_EQ(i, sortedNodes.size() - 1);
        }
    }

    std::set<size_t> indices_model_15;
    std::set<size_t> indices_model_17;
    std::set<size_t> indices_model_18;
    std::set<size_t> indices_model_19;
    std::set<size_t> indices_model_22;

    for (size_t i = 0; i < sortedNodes.size(); ++i)
    {
        if (sortedNodes[i].getName().find("model.15") != std::string::npos)
        {
            indices_model_15.insert(i);
        }
        else if (sortedNodes[i].getName().find("model.17") != std::string::npos)
        {
            indices_model_17.insert(i);
        }
        else if (sortedNodes[i].getName().find("model.18") != std::string::npos)
        {
            indices_model_18.insert(i);
        }
        else if (sortedNodes[i].getName().find("model.19") != std::string::npos)
        {
            indices_model_19.insert(i);
        }
        else if (sortedNodes[i].getName().find("model.22") != std::string::npos)
        {
            indices_model_22.insert(i);
        }
    }

    // all elements in the model_15 set should be less than all elements in the model_17 set
    for (size_t i : indices_model_15)
    {
        for (size_t j : indices_model_17)
        {
            EXPECT_LT(i, j);
        }
    }

    // all elements in the model_17 set should be less than all elements in the model_18 set
    for (size_t i : indices_model_17)
    {
        for (size_t j : indices_model_18)
        {
            EXPECT_LT(i, j);
        }
    }

    // all elements in the model_18 set should be less than all elements in the model_19 set
    for (size_t i : indices_model_18)
    {
        for (size_t j : indices_model_19)
        {
            EXPECT_LT(i, j);
        }
    }

    // all elements in the model_15 set should be less than all elements in the model_22 set
    for (size_t i : indices_model_15)
    {
        for (size_t j : indices_model_22)
        {
            EXPECT_LT(i, j);
        }
    }
}
