#include <vadugrad/nn/embedding.hpp>

#include <stdexcept>

Embedding::Embedding(int vocab_size, int d_model) : weight_(std::vector<int>{vocab_size, d_model}) {
    if (vocab_size <= 0 || d_model <= 0) {
        throw std::invalid_argument("Embedding: dimensions must be positive");
    }
}

Tensor& Embedding::weight() { return weight_; }
const Tensor& Embedding::weight() const { return weight_; }

Tensor Embedding::forward(const Tensor& token_ids) const {
    if (token_ids.ndim() != 2) {
        throw std::invalid_argument("Embedding::forward expects [B,T] token ids");
    }
    const auto s = token_ids.shape();
    const int vocab = weight_.shape()[0];
    const int dim = weight_.shape()[1];
    Tensor out({s[0], s[1], dim});
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            const int idx = static_cast<int>(token_ids({b, t}));
            if (idx < 0 || idx >= vocab) {
                throw std::invalid_argument("Embedding::forward token id out of range");
            }
            for (int d = 0; d < dim; ++d) {
                out({b, t, d}) = weight_({idx, d});
            }
        }
    }
    cache_ids_ = token_ids;
    return out;
}

Tensor Embedding::backward(const Tensor& grad_output) const {
    if (!cache_ids_) {
        throw std::invalid_argument("Embedding::backward called before forward");
    }
    if (grad_output.ndim() != 3) {
        throw std::invalid_argument("Embedding::backward expects [B,T,D]");
    }
    const auto ids_shape = cache_ids_->shape();
    const auto gs = grad_output.shape();
    if (ids_shape[0] != gs[0] || ids_shape[1] != gs[1] || gs[2] != weight_.shape()[1]) {
        throw std::invalid_argument("Embedding::backward shape mismatch");
    }
    Tensor grad_w(weight_.shape());
    grad_w.fill(0.0f);
    for (int b = 0; b < ids_shape[0]; ++b) {
        for (int t = 0; t < ids_shape[1]; ++t) {
            const int idx = static_cast<int>((*cache_ids_)({b, t}));
            for (int d = 0; d < gs[2]; ++d) {
                grad_w({idx, d}) += grad_output({b, t, d});
            }
        }
    }
    return grad_w;
}
