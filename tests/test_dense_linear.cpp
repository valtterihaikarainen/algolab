#include <gtest/gtest.h>
#include <stdexcept>
#include <vadugrad/dense_linear.hpp>

TEST(DenseLinearTest, ForwardMatchesHandComputation) {
    DenseLinear layer(2, 3);
    Tensor& W = layer.weight();
    Tensor& b = layer.bias();

    W({0, 0}) = 1.0f;
    W({0, 1}) = 2.0f;
    W({0, 2}) = 3.0f;
    W({1, 0}) = 4.0f;
    W({1, 1}) = 5.0f;
    W({1, 2}) = 6.0f;

    b({0}) = 0.1f;
    b({1}) = 0.2f;
    b({2}) = 0.3f;

    Tensor x({1, 2});
    x({0, 0}) = 1.0f;
    x({0, 1}) = 2.0f;

    Tensor y = layer.forward(x);
    ASSERT_EQ(y.shape(), (std::vector<int>{1, 3}));
    // [1,2] @ W = [9, 12, 15] + bias
    EXPECT_FLOAT_EQ(y({0, 0}), 9.1f);
    EXPECT_FLOAT_EQ(y({0, 1}), 12.2f);
    EXPECT_FLOAT_EQ(y({0, 2}), 15.3f);
}

TEST(DenseLinearTest, BackwardMatchesHandComputation) {
    DenseLinear layer(2, 3);
    Tensor& W = layer.weight();
    W({0, 0}) = 1.0f;
    W({0, 1}) = 2.0f;
    W({0, 2}) = 3.0f;
    W({1, 0}) = 4.0f;
    W({1, 1}) = 5.0f;
    W({1, 2}) = 6.0f;
    layer.bias().fill(0.0f);

    Tensor x({1, 2});
    x({0, 0}) = 1.0f;
    x({0, 1}) = 2.0f;

    Tensor grad_y({1, 3});
    grad_y({0, 0}) = 1.0f;
    grad_y({0, 1}) = 0.0f;
    grad_y({0, 2}) = 0.0f;

    const auto g = layer.backward(x, grad_y);

    EXPECT_FLOAT_EQ(g.grad_input({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(g.grad_input({0, 1}), 4.0f);

    EXPECT_FLOAT_EQ(g.grad_weight({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({1, 2}), 0.0f);

    EXPECT_FLOAT_EQ(g.grad_bias({0}), 1.0f);
    EXPECT_FLOAT_EQ(g.grad_bias({1}), 0.0f);
    EXPECT_FLOAT_EQ(g.grad_bias({2}), 0.0f);
}

TEST(DenseLinearTest, BatchedBackwardGradWeightAccumulates) {
    DenseLinear layer(2, 2);
    layer.weight().fill(0.0f);
    layer.bias().fill(0.0f);

    Tensor x({2, 2});
    x({0, 0}) = 1.0f;
    x({0, 1}) = 0.0f;
    x({1, 0}) = 0.0f;
    x({1, 1}) = 1.0f;

    Tensor grad_y({2, 2});
    grad_y({0, 0}) = 1.0f;
    grad_y({0, 1}) = 0.0f;
    grad_y({1, 0}) = 0.0f;
    grad_y({1, 1}) = 1.0f;

    const auto g = layer.backward(x, grad_y);
    // grad_w = x^T @ gy = [[1,0],[0,1]] @ [[1,0],[0,1]] = I
    EXPECT_FLOAT_EQ(g.grad_weight({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(g.grad_weight({1, 1}), 1.0f);

    EXPECT_FLOAT_EQ(g.grad_bias({0}), 1.0f);
    EXPECT_FLOAT_EQ(g.grad_bias({1}), 1.0f);
}

TEST(DenseLinearTest, InvalidFeatureCountThrows) {
    EXPECT_THROW(DenseLinear(0, 4), std::invalid_argument);
    EXPECT_THROW(DenseLinear(3, -1), std::invalid_argument);
}

TEST(DenseLinearTest, ForwardWrongInnerDimThrows) {
    DenseLinear layer(3, 2);
    Tensor x({1, 2});
    EXPECT_THROW({
        Tensor tmp = layer.forward(x);
        (void)tmp;
    }, std::invalid_argument);
}
