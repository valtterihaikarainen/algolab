#include <gtest/gtest.h>
#include <vadugrad/nn/decoder_only_transformer.hpp>
#include <vadugrad/nn/nn_ops.hpp>

TEST(DecoderOnlyTransformerTest, ForwardAndBackwardShapes) {
    DecoderOnlyTransformer model(16, 8, 2, 16, 2, 8);

    Tensor tokens({2, 4});
    tokens({0, 0}) = 1.0f;
    tokens({0, 1}) = 2.0f;
    tokens({0, 2}) = 3.0f;
    tokens({0, 3}) = 4.0f;
    tokens({1, 0}) = 5.0f;
    tokens({1, 1}) = 6.0f;
    tokens({1, 2}) = 7.0f;
    tokens({1, 3}) = 8.0f;

    const Tensor logits = model.forward(tokens);
    EXPECT_EQ(logits.shape(), (std::vector<int>{2, 4, 16}));

    Tensor targets({2, 4});
    targets({0, 0}) = 2.0f;
    targets({0, 1}) = 3.0f;
    targets({0, 2}) = 4.0f;
    targets({0, 3}) = 5.0f;
    targets({1, 0}) = 6.0f;
    targets({1, 1}) = 7.0f;
    targets({1, 2}) = 8.0f;
    targets({1, 3}) = 9.0f;

    const Tensor grad_logits = cross_entropy_grad_logits(logits, targets);
    const auto g = model.backward(grad_logits);

    EXPECT_EQ(g.grad_token_embedding.shape(), (std::vector<int>{16, 8}));
    EXPECT_EQ(g.grad_position_embedding.shape(), (std::vector<int>{8, 8}));
    EXPECT_EQ(g.grad_lm_head_weight.shape(), (std::vector<int>{8, 16}));
    EXPECT_EQ(g.grad_lm_head_bias.shape(), (std::vector<int>{16}));
}
