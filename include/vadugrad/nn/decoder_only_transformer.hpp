#pragma once

#include <optional>
#include <vector>
#include <vadugrad/core/tensor_ops.hpp>
#include <vadugrad/dense_linear.hpp>
#include <vadugrad/nn/embedding.hpp>
#include <vadugrad/nn/layer_norm.hpp>
#include <vadugrad/nn/transformer_block.hpp>

class Adam;

class DecoderOnlyTransformer {
    int vocab_size_;
    int d_model_;
    int max_seq_len_;
    Embedding token_embedding_;
    Embedding position_embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm ln_f_;
    DenseLinear lm_head_;
    mutable std::optional<Tensor> cache_tokens_;
    mutable std::optional<Tensor> cache_positions_;
    mutable std::optional<Tensor> cache_ln_out_;

public:
    DecoderOnlyTransformer(int vocab_size, int d_model, int num_heads, int d_ff, int num_layers,
                           int max_seq_len);

    [[nodiscard]] Tensor forward(const Tensor& token_ids);

    struct BackwardGradients {
        Tensor grad_token_embedding;
        Tensor grad_position_embedding;
        std::vector<TransformerBlock::BackwardGradients> block_gradients;
        LayerNorm::BackwardGradients ln_f_gradients;
        Tensor grad_lm_head_weight;
        Tensor grad_lm_head_bias;
    };

    [[nodiscard]] BackwardGradients backward(const Tensor& grad_logits);
    void initialize_parameters(float stddev = 0.02f, unsigned int seed = 42u);
    void apply_gradients(const BackwardGradients& grads, Adam& optimizer);

    Embedding& token_embedding();
    Embedding& position_embedding();
    LayerNorm& ln_f();
    DenseLinear& lm_head();
    std::vector<TransformerBlock>& blocks();
};
