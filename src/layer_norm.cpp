#include <vadugrad/nn/layer_norm.hpp>

#include <cmath>
#include <stdexcept>

LayerNorm::LayerNorm(int d_model, float eps)
    : gamma_(std::vector<int>{d_model}), beta_(std::vector<int>{d_model}), eps_(eps) {
    if (d_model <= 0) {
        throw std::invalid_argument("LayerNorm: d_model must be positive");
    }
    gamma_.fill(1.0f);
    beta_.fill(0.0f);
}

Tensor& LayerNorm::gamma() { return gamma_; }
Tensor& LayerNorm::beta() { return beta_; }
const Tensor& LayerNorm::gamma() const { return gamma_; }
const Tensor& LayerNorm::beta() const { return beta_; }

Tensor LayerNorm::forward(const Tensor& x) const {
    if (x.ndim() != 3 || x.shape()[2] != gamma_.shape()[0]) {
        throw std::invalid_argument("LayerNorm::forward expects [B,T,D] with D matching gamma");
    }
    const auto s = x.shape();
    const int bsz = s[0];
    const int tlen = s[1];
    const int dim = s[2];
    Tensor out(s);
    Tensor mean({bsz, tlen, 1});
    Tensor inv_std({bsz, tlen, 1});

    for (int b = 0; b < bsz; ++b) {
        for (int t = 0; t < tlen; ++t) {
            float mu = 0.0f;
            for (int d = 0; d < dim; ++d) {
                mu += x({b, t, d});
            }
            mu /= static_cast<float>(dim);
            float var = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float v = x({b, t, d}) - mu;
                var += v * v;
            }
            var /= static_cast<float>(dim);
            const float inv = 1.0f / std::sqrt(var + eps_);
            mean({b, t, 0}) = mu;
            inv_std({b, t, 0}) = inv;
            for (int d = 0; d < dim; ++d) {
                const float xhat = (x({b, t, d}) - mu) * inv;
                out({b, t, d}) = xhat * gamma_({d}) + beta_({d});
            }
        }
    }

    cache_input_ = x;
    cache_mean_ = mean;
    cache_inv_std_ = inv_std;
    return out;
}

LayerNorm::BackwardGradients LayerNorm::backward(const Tensor& grad_output) const {
    if (!cache_input_ || !cache_mean_ || !cache_inv_std_) {
        throw std::invalid_argument("LayerNorm::backward called before forward");
    }
    if (grad_output.shape() != cache_input_->shape()) {
        throw std::invalid_argument("LayerNorm::backward grad_output shape mismatch");
    }
    const auto s = grad_output.shape();
    const int bsz = s[0];
    const int tlen = s[1];
    const int dim = s[2];

    Tensor grad_input(s);
    Tensor grad_gamma({dim});
    Tensor grad_beta({dim});
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);

    for (int b = 0; b < bsz; ++b) {
        for (int t = 0; t < tlen; ++t) {
            const float mu = (*cache_mean_)({b, t, 0});
            const float inv = (*cache_inv_std_)({b, t, 0});

            float sum_g = 0.0f;
            float sum_gxhat = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float xhat = ((*cache_input_)({b, t, d}) - mu) * inv;
                const float g = grad_output({b, t, d}) * gamma_({d});
                sum_g += g;
                sum_gxhat += g * xhat;
            }

            for (int d = 0; d < dim; ++d) {
                const float xhat = ((*cache_input_)({b, t, d}) - mu) * inv;
                const float g = grad_output({b, t, d}) * gamma_({d});
                grad_input({b, t, d}) =
                    inv * (g - sum_g / static_cast<float>(dim) -
                           xhat * sum_gxhat / static_cast<float>(dim));
                grad_gamma({d}) += grad_output({b, t, d}) * xhat;
                grad_beta({d}) += grad_output({b, t, d});
            }
        }
    }

    return BackwardGradients{grad_input, grad_gamma, grad_beta};
}
