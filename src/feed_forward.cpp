#include <vadugrad/nn/feed_forward.hpp>

#include <stdexcept>
#include <vadugrad/nn/activations.hpp>

FeedForward::FeedForward(int d_model, int d_ff) : fc1_(d_model, d_ff), fc2_(d_ff, d_model) {
    if (d_model <= 0 || d_ff <= 0) {
        throw std::invalid_argument("FeedForward: dimensions must be positive");
    }
}

DenseLinear& FeedForward::fc1() { return fc1_; }
DenseLinear& FeedForward::fc2() { return fc2_; }
const DenseLinear& FeedForward::fc1() const { return fc1_; }
const DenseLinear& FeedForward::fc2() const { return fc2_; }

Tensor FeedForward::forward(const Tensor& x) const {
    if (x.ndim() != 3 || x.shape()[2] != fc1_.in_features()) {
        throw std::invalid_argument("FeedForward::forward expects [B,T,d_model]");
    }
    const auto s = x.shape();
    const Tensor x2d = flatten_bt(x);
    const Tensor h_pre_2d = fc1_.forward(x2d);
    const Tensor h_pre = unflatten_bt(h_pre_2d, s[0], s[1], fc1_.out_features());
    const Tensor h = gelu(h_pre);
    const Tensor y2d = fc2_.forward(flatten_bt(h));
    const Tensor y = unflatten_bt(y2d, s[0], s[1], fc2_.out_features());
    cache_input_ = x;
    cache_preact_ = h_pre;
    return y;
}

FeedForward::BackwardGradients FeedForward::backward(const Tensor& grad_output) const {
    if (!cache_input_ || !cache_preact_) {
        throw std::invalid_argument("FeedForward::backward called before forward");
    }
    if (grad_output.shape() != cache_input_->shape()) {
        throw std::invalid_argument("FeedForward::backward grad_output shape mismatch");
    }
    const auto s = grad_output.shape();
    const Tensor h = gelu(*cache_preact_);
    const auto g2 = fc2_.backward(flatten_bt(h), flatten_bt(grad_output));
    const Tensor grad_h = unflatten_bt(g2.grad_input, s[0], s[1], fc1_.out_features());
    const Tensor grad_h_pre = gelu_backward(*cache_preact_, grad_h);
    const auto g1 = fc1_.backward(flatten_bt(*cache_input_), flatten_bt(grad_h_pre));
    return BackwardGradients{unflatten_bt(g1.grad_input, s[0], s[1], fc1_.in_features()),
                             g1.grad_weight, g1.grad_bias, g2.grad_weight, g2.grad_bias};
}
