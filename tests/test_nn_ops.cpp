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
