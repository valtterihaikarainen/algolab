#include <gtest/gtest.h>
#include <vector>
#include <vadugrad/tensor.hpp>

TEST(ComputeStridesTest, Basic3D) {
    std::vector<int> shape = {2, 3, 4};
    auto strides = compute_strides(shape);

    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(ComputeStridesTest, SingleDimension) {
    std::vector<int> shape = {5};
    auto strides = compute_strides(shape);

    ASSERT_EQ(strides.size(), 1u);
    EXPECT_EQ(strides[0], 1);
}

TEST(TensorTest, CopyConstructorDoesNotCrash) {
    std::vector<int> shape = {2, 3, 4};
    Tensor a(shape);
    Tensor b = a;  // copy constructor

    SUCCEED();
}

TEST(TensorTest, AssignmentOperatorDoesNotCrash) {
    std::vector<int> shape = {2, 3, 4};
    Tensor a(shape);
    Tensor b(shape);

    b = a;  // assignment operator

    SUCCEED();
}


