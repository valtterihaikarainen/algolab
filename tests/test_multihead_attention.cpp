#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include <vadugrad/multihead_attention.hpp>

TEST(MultiHeadAttentionTest, ConstructorValidation) {
    EXPECT_THROW(MultiHeadAttention(0, 2), std::invalid_argument);
    EXPECT_THROW(MultiHeadAttention(8, 0), std::invalid_argument);
    EXPECT_THROW(MultiHeadAttention(7, 2), std::invalid_argument);
}

TEST(MultiHeadAttentionTest, ForwardSelfAndCrossShapes) {
    MultiHeadAttention mha(4, 2, false);
    Tensor q({2, 3, 4});
    Tensor kv({2, 5, 4});

    const Tensor y_cross = mha.forward(q, kv);
    EXPECT_EQ(y_cross.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(mha.attention_probs().shape(), (std::vector<int>{2, 2, 3, 5}));

    const Tensor y_self = mha.forward(q);
    EXPECT_EQ(y_self.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(mha.attention_probs().shape(), (std::vector<int>{2, 2, 3, 3}));
}

TEST(MultiHeadAttentionTest, BackwardShapesForCrossAttention) {
    MultiHeadAttention mha(4, 2, false);
    Tensor q({2, 3, 4});
    Tensor kv({2, 5, 4});
    (void)mha.forward(q, kv);

    Tensor grad_out({2, 3, 4});
    grad_out.fill(1.0f);
    const auto g = mha.backward(grad_out);

    EXPECT_EQ(g.grad_q_input.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(g.grad_kv_input.shape(), (std::vector<int>{2, 5, 4}));
    EXPECT_EQ(g.grad_Wq.shape(), (std::vector<int>{4, 4}));
    EXPECT_EQ(g.grad_bq.shape(), (std::vector<int>{4}));
    EXPECT_EQ(g.grad_Wk.shape(), (std::vector<int>{4, 4}));
    EXPECT_EQ(g.grad_bk.shape(), (std::vector<int>{4}));
    EXPECT_EQ(g.grad_Wv.shape(), (std::vector<int>{4, 4}));
    EXPECT_EQ(g.grad_bv.shape(), (std::vector<int>{4}));
    EXPECT_EQ(g.grad_Wo.shape(), (std::vector<int>{4, 4}));
    EXPECT_EQ(g.grad_bo.shape(), (std::vector<int>{4}));
}

TEST(MultiHeadAttentionTest, BackwardBeforeForwardThrows) {
    MultiHeadAttention mha(4, 2, false);
    Tensor grad_out({1, 1, 4});
    EXPECT_THROW({
        auto tmp = mha.backward(grad_out);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(MultiHeadAttentionTest, CausalAttentionMasksFutureInSelfAttention) {
    MultiHeadAttention mha(4, 2, true);
    Tensor x({1, 3, 4});
    (void)mha.forward(x);

    const Tensor& p = mha.attention_probs();
    EXPECT_FLOAT_EQ(p({0, 0, 0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(p({0, 0, 0, 2}), 0.0f);
    EXPECT_FLOAT_EQ(p({0, 0, 1, 2}), 0.0f);
}
