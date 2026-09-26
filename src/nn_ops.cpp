#include <vadugrad/nn/nn_ops.hpp>

#include <cmath>
#include <stdexcept>

namespace {
void check_rank3(const Tensor& x, const char* name) {
    if (x.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be rank-3 [B,T,V]");
    }
}
}  // namespace

Tensor softmax_last_dim(const Tensor& x) {
    check_rank3(x, "softmax_last_dim input");
    const auto s = x.shape();
    Tensor out(s);
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            float max_v = -1e30f;
            for (int v = 0; v < s[2]; ++v) {
                const float val = x({b, t, v});
                if (val > max_v) {
                    max_v = val;
                }
            }
            float denom = 0.0f;
            for (int v = 0; v < s[2]; ++v) {
                const float e = std::exp(x({b, t, v}) - max_v);
                out({b, t, v}) = e;
                denom += e;
            }
            for (int v = 0; v < s[2]; ++v) {
                out({b, t, v}) /= denom;
            }
        }
    }
    return out;
}

Tensor log_softmax_last_dim(const Tensor& x) {
    check_rank3(x, "log_softmax_last_dim input");
    const auto s = x.shape();
    Tensor out(s);
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            float max_v = -1e30f;
            for (int v = 0; v < s[2]; ++v) {
                const float val = x({b, t, v});
                if (val > max_v) {
                    max_v = val;
                }
            }
            float sum_exp = 0.0f;
            for (int v = 0; v < s[2]; ++v) {
                sum_exp += std::exp(x({b, t, v}) - max_v);
            }
            const float log_denom = max_v + std::log(sum_exp);
            for (int v = 0; v < s[2]; ++v) {
                out({b, t, v}) = x({b, t, v}) - log_denom;
            }
        }
    }
    return out;
}

float cross_entropy_mean(const Tensor& logits, const Tensor& target_ids, int ignore_index) {
    check_rank3(logits, "cross_entropy_mean logits");
    if (target_ids.ndim() != 2) {
        throw std::invalid_argument("cross_entropy_mean target_ids must be rank-2 [B,T]");
    }
    const auto ls = logits.shape();
    const auto ts = target_ids.shape();
    if (ts[0] != ls[0] || ts[1] != ls[1]) {
        throw std::invalid_argument("cross_entropy_mean target shape mismatch");
    }

    const Tensor logp = log_softmax_last_dim(logits);
    float loss = 0.0f;
    int count = 0;
    for (int b = 0; b < ls[0]; ++b) {
        for (int t = 0; t < ls[1]; ++t) {
            const int y = static_cast<int>(target_ids({b, t}));
            if (y == ignore_index) {
                continue;
            }
            if (y < 0 || y >= ls[2]) {
                throw std::invalid_argument("cross_entropy_mean target id out of range");
            }
            loss -= logp({b, t, y});
            ++count;
        }
    }
    if (count == 0) {
        return 0.0f;
    }
    return loss / static_cast<float>(count);
}

Tensor cross_entropy_grad_logits(const Tensor& logits, const Tensor& target_ids, int ignore_index) {
    check_rank3(logits, "cross_entropy_grad_logits logits");
    if (target_ids.ndim() != 2) {
        throw std::invalid_argument("cross_entropy_grad_logits target_ids must be rank-2 [B,T]");
    }
    const auto ls = logits.shape();
    const auto ts = target_ids.shape();
    if (ts[0] != ls[0] || ts[1] != ls[1]) {
        throw std::invalid_argument("cross_entropy_grad_logits target shape mismatch");
    }
    Tensor grad = softmax_last_dim(logits);
    int count = 0;
    for (int b = 0; b < ls[0]; ++b) {
        for (int t = 0; t < ls[1]; ++t) {
            const int y = static_cast<int>(target_ids({b, t}));
            if (y == ignore_index) {
                for (int v = 0; v < ls[2]; ++v) {
                    grad({b, t, v}) = 0.0f;
                }
                continue;
            }
            if (y < 0 || y >= ls[2]) {
                throw std::invalid_argument("cross_entropy_grad_logits target id out of range");
            }
            grad({b, t, y}) -= 1.0f;
            ++count;
        }
    }
    if (count == 0) {
        return grad;
    }
    const float inv = 1.0f / static_cast<float>(count);
    for (int i = 0; i < grad.numel(); ++i) {
        grad.data()[i] *= inv;
    }
    return grad;
}
