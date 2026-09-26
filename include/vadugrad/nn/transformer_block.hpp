#pragma once

#include <optional>
#include <vadugrad/core/tensor_ops.hpp>
#include <vadugrad/multihead_attention.hpp>
#include <vadugrad/nn/feed_forward.hpp>
#include <vadugrad/nn/layer_norm.hpp>

class TransformerBlock {
    LayerNorm ln1_;
    MultiHeadAttention mha_;
    LayerNorm ln2_;
    FeedForward ffn_;
    mutable std::optional<Tensor> cache_input_;
    mutable std::optional<Tensor> cache_post_attn_;

public:
    TransformerBlock(int d_model, int num_heads, int d_ff, bool causal);

    LayerNorm& ln1();
    LayerNorm& ln2();
    MultiHeadAttention& mha();
    FeedForward& ffn();

    const LayerNorm& ln1() const;
    const LayerNorm& ln2() const;
    const MultiHeadAttention& mha() const;
    const FeedForward& ffn() const;

    [[nodiscard]] Tensor forward(const Tensor& x);

    struct BackwardGradients {
        Tensor grad_input;
        LayerNorm::BackwardGradients ln1_gradients;
        MultiHeadAttention::BackwardGradients mha_gradients;
        LayerNorm::BackwardGradients ln2_gradients;
        FeedForward::BackwardGradients ffn_gradients;
    };

    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output);
};
