#include <vadugrad/nn/transformer_block.hpp>

#include <stdexcept>

TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff, bool causal)
    : ln1_(d_model), mha_(d_model, num_heads, causal), ln2_(d_model), ffn_(d_model, d_ff) {}

LayerNorm& TransformerBlock::ln1() { return ln1_; }
LayerNorm& TransformerBlock::ln2() { return ln2_; }
MultiHeadAttention& TransformerBlock::mha() { return mha_; }
FeedForward& TransformerBlock::ffn() { return ffn_; }
const LayerNorm& TransformerBlock::ln1() const { return ln1_; }
const LayerNorm& TransformerBlock::ln2() const { return ln2_; }
const MultiHeadAttention& TransformerBlock::mha() const { return mha_; }
const FeedForward& TransformerBlock::ffn() const { return ffn_; }

Tensor TransformerBlock::forward(const Tensor& x) {
    const Tensor n1 = ln1_.forward(x);
    const Tensor a = mha_.forward(n1);
    const Tensor x2 = add3d(x, a);
    const Tensor n2 = ln2_.forward(x2);
    const Tensor f = ffn_.forward(n2);
    const Tensor y = add3d(x2, f);
    cache_input_ = x;
    cache_post_attn_ = x2;
    return y;
}

TransformerBlock::BackwardGradients TransformerBlock::backward(const Tensor& grad_output) {
    if (!cache_input_ || !cache_post_attn_) {
        throw std::invalid_argument("TransformerBlock::backward called before forward");
    }
    if (grad_output.shape() != cache_input_->shape()) {
        throw std::invalid_argument("TransformerBlock::backward grad_output shape mismatch");
    }

    const Tensor grad_x2_from_residual = grad_output;
    const auto ffg = ffn_.backward(grad_output);
    const auto ln2g = ln2_.backward(ffg.grad_input);
    const Tensor grad_x2 = add3d(grad_x2_from_residual, ln2g.grad_input);
    const Tensor grad_x_from_residual = grad_x2;
    const auto mhag = mha_.backward(grad_x2);
    const Tensor grad_ln1 = add3d(mhag.grad_q_input, mhag.grad_kv_input);
    const auto ln1g = ln1_.backward(grad_ln1);
    const Tensor grad_x = add3d(grad_x_from_residual, ln1g.grad_input);

    return BackwardGradients{grad_x, ln1g, mhag, ln2g, ffg};
}
