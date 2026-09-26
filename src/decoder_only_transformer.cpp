#include <vadugrad/nn/decoder_only_transformer.hpp>

#include <stdexcept>
#include <vadugrad/nn/parameter_init.hpp>
#include <vadugrad/optim/adam.hpp>

DecoderOnlyTransformer::DecoderOnlyTransformer(int vocab_size, int d_model, int num_heads, int d_ff,
                                               int num_layers, int max_seq_len)
    : vocab_size_(vocab_size),
      d_model_(d_model),
      max_seq_len_(max_seq_len),
      token_embedding_(vocab_size, d_model),
      position_embedding_(max_seq_len, d_model),
      ln_f_(d_model),
      lm_head_(d_model, vocab_size) {
    if (vocab_size <= 0 || d_model <= 0 || num_heads <= 0 || d_ff <= 0 || num_layers <= 0 ||
        max_seq_len <= 0) {
        throw std::invalid_argument("DecoderOnlyTransformer: all dimensions must be positive");
    }
    blocks_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(d_model, num_heads, d_ff, true);
    }
}

Tensor DecoderOnlyTransformer::forward(const Tensor& token_ids) {
    if (token_ids.ndim() != 2) {
        throw std::invalid_argument("DecoderOnlyTransformer::forward expects token_ids [B,T]");
    }
    const auto s = token_ids.shape();
    if (s[1] > max_seq_len_) {
        throw std::invalid_argument("DecoderOnlyTransformer::forward sequence exceeds max_seq_len");
    }
    Tensor pos_ids({s[0], s[1]});
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            pos_ids({b, t}) = static_cast<float>(t);
        }
    }

    Tensor x = add3d(token_embedding_.forward(token_ids), position_embedding_.forward(pos_ids));
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    const Tensor ln_out = ln_f_.forward(x);
    const Tensor logits2d = lm_head_.forward(flatten_bt(ln_out));
    const Tensor logits = unflatten_bt(logits2d, s[0], s[1], vocab_size_);

    cache_tokens_ = token_ids;
    cache_positions_ = pos_ids;
    cache_ln_out_ = ln_out;
    return logits;
}

DecoderOnlyTransformer::BackwardGradients DecoderOnlyTransformer::backward(const Tensor& grad_logits) {
    if (!cache_tokens_ || !cache_positions_ || !cache_ln_out_) {
        throw std::invalid_argument("DecoderOnlyTransformer::backward called before forward");
    }
    if (grad_logits.ndim() != 3) {
        throw std::invalid_argument("DecoderOnlyTransformer::backward expects [B,T,V]");
    }
    const auto s = grad_logits.shape();
    const auto lg = lm_head_.backward(flatten_bt(*cache_ln_out_), flatten_bt(grad_logits));
    Tensor grad_x = unflatten_bt(lg.grad_input, s[0], s[1], d_model_);
    const auto lnf = ln_f_.backward(grad_x);
    grad_x = lnf.grad_input;

    std::vector<TransformerBlock::BackwardGradients> bgrads;
    bgrads.reserve(blocks_.size());
    std::vector<TransformerBlock::BackwardGradients> rev_bgrads;
    rev_bgrads.reserve(blocks_.size());
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        auto bg = blocks_[i].backward(grad_x);
        grad_x = bg.grad_input;
        rev_bgrads.push_back(bg);
    }
    for (int i = static_cast<int>(rev_bgrads.size()) - 1; i >= 0; --i) {
        bgrads.push_back(rev_bgrads[static_cast<std::size_t>(i)]);
    }

    const Tensor grad_tok = token_embedding_.backward(grad_x);
    const Tensor grad_pos = position_embedding_.backward(grad_x);

    return BackwardGradients{grad_tok, grad_pos, bgrads, lnf, lg.grad_weight, lg.grad_bias};
}

Embedding& DecoderOnlyTransformer::token_embedding() { return token_embedding_; }
Embedding& DecoderOnlyTransformer::position_embedding() { return position_embedding_; }
LayerNorm& DecoderOnlyTransformer::ln_f() { return ln_f_; }
DenseLinear& DecoderOnlyTransformer::lm_head() { return lm_head_; }
std::vector<TransformerBlock>& DecoderOnlyTransformer::blocks() { return blocks_; }

void DecoderOnlyTransformer::initialize_parameters(float stddev, unsigned int seed) {
    initialize_tensor_normal(token_embedding_.weight(), stddev, seed);
    initialize_tensor_normal(position_embedding_.weight(), stddev, seed);
    initialize_tensor_normal(lm_head_.weight(), stddev, seed);
    lm_head_.bias().fill(0.0f);
    ln_f_.gamma().fill(1.0f);
    ln_f_.beta().fill(0.0f);

    for (auto& block : blocks_) {
        block.ln1().gamma().fill(1.0f);
        block.ln1().beta().fill(0.0f);
        block.ln2().gamma().fill(1.0f);
        block.ln2().beta().fill(0.0f);

        initialize_tensor_normal(block.mha().q_proj().weight(), stddev, seed);
        initialize_tensor_normal(block.mha().k_proj().weight(), stddev, seed);
        initialize_tensor_normal(block.mha().v_proj().weight(), stddev, seed);
        initialize_tensor_normal(block.mha().o_proj().weight(), stddev, seed);
        block.mha().q_proj().bias().fill(0.0f);
        block.mha().k_proj().bias().fill(0.0f);
        block.mha().v_proj().bias().fill(0.0f);
        block.mha().o_proj().bias().fill(0.0f);

        initialize_tensor_normal(block.ffn().fc1().weight(), stddev, seed);
        initialize_tensor_normal(block.ffn().fc2().weight(), stddev, seed);
        block.ffn().fc1().bias().fill(0.0f);
        block.ffn().fc2().bias().fill(0.0f);
    }
}

void DecoderOnlyTransformer::apply_gradients(const BackwardGradients& grads, Adam& optimizer) {
    optimizer.step(token_embedding_.weight(), grads.grad_token_embedding);
    optimizer.step(position_embedding_.weight(), grads.grad_position_embedding);
    optimizer.step(ln_f_.gamma(), grads.ln_f_gradients.grad_gamma);
    optimizer.step(ln_f_.beta(), grads.ln_f_gradients.grad_beta);
    optimizer.step(lm_head_.weight(), grads.grad_lm_head_weight);
    optimizer.step(lm_head_.bias(), grads.grad_lm_head_bias);

    if (grads.block_gradients.size() != blocks_.size()) {
        throw std::invalid_argument("apply_gradients: block gradients size mismatch");
    }
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        auto& block = blocks_[i];
        const auto& bg = grads.block_gradients[i];

        optimizer.step(block.ln1().gamma(), bg.ln1_gradients.grad_gamma);
        optimizer.step(block.ln1().beta(), bg.ln1_gradients.grad_beta);
        optimizer.step(block.mha().q_proj().weight(), bg.mha_gradients.grad_Wq);
        optimizer.step(block.mha().q_proj().bias(), bg.mha_gradients.grad_bq);
        optimizer.step(block.mha().k_proj().weight(), bg.mha_gradients.grad_Wk);
        optimizer.step(block.mha().k_proj().bias(), bg.mha_gradients.grad_bk);
        optimizer.step(block.mha().v_proj().weight(), bg.mha_gradients.grad_Wv);
        optimizer.step(block.mha().v_proj().bias(), bg.mha_gradients.grad_bv);
        optimizer.step(block.mha().o_proj().weight(), bg.mha_gradients.grad_Wo);
        optimizer.step(block.mha().o_proj().bias(), bg.mha_gradients.grad_bo);
        optimizer.step(block.ln2().gamma(), bg.ln2_gradients.grad_gamma);
        optimizer.step(block.ln2().beta(), bg.ln2_gradients.grad_beta);
        optimizer.step(block.ffn().fc1().weight(), bg.ffn_gradients.grad_fc1_weight);
        optimizer.step(block.ffn().fc1().bias(), bg.ffn_gradients.grad_fc1_bias);
        optimizer.step(block.ffn().fc2().weight(), bg.ffn_gradients.grad_fc2_weight);
        optimizer.step(block.ffn().fc2().bias(), bg.ffn_gradients.grad_fc2_bias);
    }
}
