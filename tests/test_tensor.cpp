#include <gtest/gtest.h>
#include <vector>
#include <vadugrad/tensor.hpp>
#include <stdexcept>

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

TEST(TensorOpsTest, ElementwiseTensorTensorOps) {
    Tensor a({2, 2});
    Tensor b({2, 2});

    a({0, 0}) = 1.0f; a({0, 1}) = 2.0f;
    a({1, 0}) = 3.0f; a({1, 1}) = 4.0f;

    b({0, 0}) = 5.0f; b({0, 1}) = 6.0f;
    b({1, 0}) = 7.0f; b({1, 1}) = 8.0f;

    Tensor c_add = add(a, b);
    Tensor c_sub = sub(a, b);
    Tensor c_mul = mul(a, b);
    Tensor c_div = div(b, a);

    EXPECT_FLOAT_EQ(c_add({1, 1}), 12.0f);
    EXPECT_FLOAT_EQ(c_sub({0, 0}), -4.0f);
    EXPECT_FLOAT_EQ(c_mul({1, 0}), 21.0f);
    EXPECT_FLOAT_EQ(c_div({1, 1}), 2.0f);
}

TEST(TensorOpsTest, ElementwiseShapeMismatchThrows) {
    Tensor a({2, 2});
    Tensor b({2, 3});
    EXPECT_THROW({
        Tensor tmp = add(a, b);
        (void)tmp;
    }, std::invalid_argument);
    EXPECT_THROW({
        Tensor tmp = sub(a, b);
        (void)tmp;
    }, std::invalid_argument);
    EXPECT_THROW({
        Tensor tmp = mul(a, b);
        (void)tmp;
    }, std::invalid_argument);
    EXPECT_THROW({
        Tensor tmp = div(a, b);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorOpsTest, ScalarOpsAndOperatorsWork) {
    Tensor a({2, 2});
    a({0, 0}) = 2.0f; a({0, 1}) = 4.0f;
    a({1, 0}) = 6.0f; a({1, 1}) = 8.0f;

    Tensor c1 = add(a, 1.0f);
    Tensor c2 = sub(a, 1.0f);
    Tensor c3 = mul(a, 2.0f);
    Tensor c4 = div(a, 2.0f);
    Tensor c5 = a * 3.0f;
    Tensor c6 = 3.0f * a;

    EXPECT_FLOAT_EQ(c1({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c2({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(c3({1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(c4({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(c5({0, 1}), 12.0f);
    EXPECT_FLOAT_EQ(c6({1, 1}), 24.0f);
}

TEST(TensorOpsTest, Transpose2DWorks) {
    Tensor x({2, 3});
    x({0, 0}) = 1.0f; x({0, 1}) = 2.0f; x({0, 2}) = 3.0f;
    x({1, 0}) = 4.0f; x({1, 1}) = 5.0f; x({1, 2}) = 6.0f;

    Tensor t = transpose2d(x);
    EXPECT_EQ(t.shape(), (std::vector<int>{3, 2}));
    EXPECT_FLOAT_EQ(t({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t({2, 0}), 3.0f);
}

TEST(TensorOpsTest, Transpose2DNonRank2Throws) {
    Tensor x({2, 2, 2});
    EXPECT_THROW({
        Tensor tmp = transpose2d(x);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorOpsTest, MatmulWorks) {
    Tensor a({2, 3});
    Tensor b({3, 2});

    // a = [[1,2,3],[4,5,6]]
    a({0, 0}) = 1.0f; a({0, 1}) = 2.0f; a({0, 2}) = 3.0f;
    a({1, 0}) = 4.0f; a({1, 1}) = 5.0f; a({1, 2}) = 6.0f;

    // b = [[7,8],[9,10],[11,12]]
    b({0, 0}) = 7.0f;  b({0, 1}) = 8.0f;
    b({1, 0}) = 9.0f;  b({1, 1}) = 10.0f;
    b({2, 0}) = 11.0f; b({2, 1}) = 12.0f;

    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int>{2, 2}));
    EXPECT_FLOAT_EQ(c({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c({1, 1}), 154.0f);
}

TEST(TensorOpsTest, MatmulBadShapeThrows) {
    Tensor a({2, 3});
    Tensor b({4, 2});
    EXPECT_THROW({
        Tensor tmp = matmul(a, b);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorOpsTest, SumAxisKeepdimFalseAndTrue) {
    Tensor x({2, 3});
    x({0, 0}) = 1.0f; x({0, 1}) = 2.0f; x({0, 2}) = 3.0f;
    x({1, 0}) = 4.0f; x({1, 1}) = 5.0f; x({1, 2}) = 6.0f;

    Tensor s0 = sum(x, 0, false);  // [3]
    EXPECT_EQ(s0.shape(), (std::vector<int>{3}));
    EXPECT_FLOAT_EQ(s0({0}), 5.0f);
    EXPECT_FLOAT_EQ(s0({1}), 7.0f);
    EXPECT_FLOAT_EQ(s0({2}), 9.0f);

    Tensor s1k = sum(x, 1, true);  // [2,1]
    EXPECT_EQ(s1k.shape(), (std::vector<int>{2, 1}));
    EXPECT_FLOAT_EQ(s1k({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(s1k({1, 0}), 15.0f);
}

TEST(TensorOpsTest, SumBadAxisThrows) {
    Tensor x({2, 2});
    EXPECT_THROW({
        Tensor tmp = sum(x, -1);
        (void)tmp;
    }, std::invalid_argument);
    EXPECT_THROW({
        Tensor tmp = sum(x, 2);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorTest, FillAndValuesPreservedOnCopy) {
    Tensor a({2, 2});
    a.fill(3.5f);
    Tensor b = a;
    EXPECT_FLOAT_EQ(b({0, 0}), 3.5f);
    EXPECT_FLOAT_EQ(b({1, 1}), 3.5f);
    b({0, 0}) = 1.0f;
    EXPECT_FLOAT_EQ(a({0, 0}), 3.5f);
}

TEST(TensorTest, ReshapePreservesDataOrder) {
    Tensor t({2, 3});
    for (int i = 0; i < 6; ++i) {
        t.data()[i] = static_cast<float>(i + 1);
    }
    t.reshape({3, 2});
    EXPECT_EQ(t.shape(), (std::vector<int>{3, 2}));
    // Row-major: [1,2,3,4,5,6] -> 3x2 [[1,2],[3,4],[5,6]]
    EXPECT_FLOAT_EQ(t({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t({2, 1}), 6.0f);
}

TEST(TensorTest, ReshapeInvalidThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
    EXPECT_THROW(t.reshape({2, 0}), std::invalid_argument);
}

TEST(TensorTest, OffsetOutOfRangeThrows) {
    Tensor t({2, 2});
    EXPECT_THROW({
        (void)t.at(std::vector<int>{2, 0});
    }, std::out_of_range);
    EXPECT_THROW({
        (void)t.at(std::vector<int>{0, -1});
    }, std::out_of_range);
}

TEST(ComputeStridesTest, EmptyShapeThrows) {
    EXPECT_THROW(compute_strides({}), std::invalid_argument);
}
