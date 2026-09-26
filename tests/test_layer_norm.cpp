#include <gtest/gtest.h>
#include <vadugrad/nn/layer_norm.hpp>

TEST(LayerNormTest, ForwardBackwardShapes) {
    LayerNorm ln(4);
    Tensor x({2, 3, 4});
    x.fill(1.0f);
    const Tensor y = ln.forward(x);
    EXPECT_EQ(y.shape(), (std::vector<int>{2, 3, 4}));

    Tensor grad_y({2, 3, 4});
    grad_y.fill(1.0f);
    const auto g = ln.backward(grad_y);
    EXPECT_EQ(g.grad_input.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(g.grad_gamma.shape(), (std::vector<int>{4}));
    EXPECT_EQ(g.grad_beta.shape(), (std::vector<int>{4}));
}
