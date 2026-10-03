#include <gtest/gtest.h>
#include <vadugrad/nn/nn_ops.hpp>

TEST(NnOpsTest, CrossEntropyLossAndGradShape) {
    Tensor logits({1, 2, 3});
    logits({0, 0, 0}) = 1.0f;
    logits({0, 0, 1}) = 2.0f;
    logits({0, 0, 2}) = 0.0f;
    logits({0, 1, 0}) = -1.0f;
    logits({0, 1, 1}) = 0.0f;
    logits({0, 1, 2}) = 1.0f;

    Tensor target({1, 2});
    target({0, 0}) = 1.0f;
    target({0, 1}) = 2.0f;

    const float loss = cross_entropy_mean(logits, target);
    EXPECT_GT(loss, 0.0f);
    const Tensor grad = cross_entropy_grad_logits(logits, target);
    EXPECT_EQ(grad.shape(), (std::vector<int>{1, 2, 3}));
}

TEST(NnOpsTest, RowSoftmaxAndCrossEntropy) {
    Tensor logits({2, 3});
    logits({0, 0}) = 1.0f;
    logits({0, 1}) = 2.0f;
    logits({0, 2}) = 0.0f;
    logits({1, 0}) = 0.0f;
    logits({1, 1}) = 0.0f;
    logits({1, 2}) = 0.0f;

    Tensor labels({2, 1});
    labels({0, 0}) = 1.0f;
    labels({1, 0}) = 2.0f;

    const Tensor p = softmax_rows(logits);
    EXPECT_NEAR(p({0, 0}) + p({0, 1}) + p({0, 2}), 1.0f, 1e-5f);

    const float loss = cross_entropy_mean_rows(logits, labels);
    EXPECT_GT(loss, 0.0f);

    const Tensor g = cross_entropy_grad_logits_rows(logits, labels);
    EXPECT_EQ(g.shape(), (std::vector<int>{2, 3}));
}
