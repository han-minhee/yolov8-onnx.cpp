#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "parser/npy_parser.hpp"

TEST(NPYTest, NPYTest_0)
{
    EXPECT_EQ(0, 0);
}

TEST(NPYTest, NPYTest_1)
{
    EXPECT_EQ(1, 1);
}

TEST(NPYTest, NPYTest_NPYTest_Parse_Float32_Dim_Test)
{
    Tensor t = NpyParser::load("../tests/data/npy/images.npy");
    EXPECT_EQ(t.getNDim(), 4);
    std::vector<size_t> dims = t.getDims();
    EXPECT_EQ(dims.size(), 4);
    EXPECT_EQ(dims[0], 1);
    EXPECT_EQ(dims[1], 3);
    EXPECT_EQ(dims[2], 640);
    EXPECT_EQ(dims[3], 640);
}

TEST(NPYTest, NPYTest_NPYTest_Parse_Float32_Stride_Test)
{
    Tensor t = NpyParser::load("../tests/data/npy/images.npy");
    std::vector<size_t> strides = t.getStrides();
    EXPECT_EQ(strides.size(), 4);
    EXPECT_EQ(strides[0], 1228800);
    EXPECT_EQ(strides[1], 409600);
    EXPECT_EQ(strides[2], 640);
    EXPECT_EQ(strides[3], 1);
}