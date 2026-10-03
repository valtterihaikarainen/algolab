/**
 * @file butterfly_linear.cpp
 * @brief Implementation of @ref ButterflyLinear.
 */

#include <vadugrad/butterfly_linear.hpp>

#include <stdexcept>

namespace {

bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

int log2_int(int n) {
    int l = 0;
    int v = n;
    while (v > 1) {
        v >>= 1;
        ++l;
    }
    return l;
}

int pair_index_for_stride(int i, int stride) {
    // Unique index in [0, n/2) for the unordered pair {i, i^stride} when stride is a power of two.
    if (!is_power_of_two(stride) || stride <= 0) {
        throw std::invalid_argument("pair_index_for_stride: stride must be a positive power of two");
    }
    const int stage = log2_int(stride);
    const int low = i & (stride - 1);
    const int high = i >> (stage + 1);
    return (high << stage) | low;
}

void validate_rank2_inner(const Tensor& x, int n, const char* name) {
    if (x.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be rank-2 [batch, n]");
    }
    if (x.shape()[1] != n) {
        throw std::invalid_argument(std::string(name) + " inner dimension must equal n");
    }
}

}  // namespace

ButterflyLinear::ButterflyLinear(int n) : n_(n), num_stages_(log2_int(n)) {
    if (n < 2 || !is_power_of_two(n)) {
        throw std::invalid_argument("ButterflyLinear: n must be a power of two and >= 2");
    }
    stage_weights_.reserve(num_stages_);
    for (int s = 0; s < num_stages_; ++s) {
        stage_weights_.emplace_back(std::vector<int>{n / 2, 2, 2});
    }
}

int ButterflyLinear::dim() const { return n_; }
int ButterflyLinear::num_stages() const { return num_stages_; }

Tensor& ButterflyLinear::stage_weight(int s) { return stage_weights_.at(static_cast<std::size_t>(s)); }
const Tensor& ButterflyLinear::stage_weight(int s) const {
    return stage_weights_.at(static_cast<std::size_t>(s));
}

Tensor ButterflyLinear::forward(const Tensor& x) const {
    validate_rank2_inner(x, n_, "ButterflyLinear::forward x");

    const int batch = x.shape()[0];
    Tensor y(x.shape());
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            y({b, i}) = x({b, i});
        }
    }

    std::vector<Tensor> stage_inputs;
    stage_inputs.reserve(static_cast<std::size_t>(num_stages_));

    for (int s = 0; s < num_stages_; ++s) {
        stage_inputs.push_back(y);
        const int stride = 1 << s;
        Tensor next(y.shape());
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n_; ++i) {
                const int j = i ^ stride;
                if (j < i) {
                    continue;
                }

                const int k = pair_index_for_stride(i, stride);

                const float xi = y({b, i});
                const float xj = y({b, j});

                const float w00 = stage_weights_[static_cast<std::size_t>(s)]({k, 0, 0});
                const float w01 = stage_weights_[static_cast<std::size_t>(s)]({k, 0, 1});
                const float w10 = stage_weights_[static_cast<std::size_t>(s)]({k, 1, 0});
                const float w11 = stage_weights_[static_cast<std::size_t>(s)]({k, 1, 1});

                next({b, i}) = w00 * xi + w01 * xj;
                next({b, j}) = w10 * xi + w11 * xj;
            }
        }
        y = next;
    }

    cache_stage_inputs_ = std::move(stage_inputs);
    return y;
}

ButterflyLinear::BackwardGradients ButterflyLinear::backward(const Tensor& grad_output) const {
    if (!cache_stage_inputs_) {
        throw std::invalid_argument("ButterflyLinear::backward called before forward");
    }
    validate_rank2_inner(grad_output, n_, "ButterflyLinear::backward grad_output");

    const int batch = grad_output.shape()[0];
    const auto& stage_inputs = *cache_stage_inputs_;
    if (static_cast<int>(stage_inputs.size()) != num_stages_) {
        throw std::invalid_argument("ButterflyLinear::backward: internal cache corrupted");
    }

    std::vector<Tensor> grad_stages;
    grad_stages.reserve(stage_weights_.size());
    for (const auto& w : stage_weights_) {
        grad_stages.emplace_back(w.shape());
        grad_stages.back().fill(0.0f);
    }

    Tensor gy = grad_output;
    for (int s = num_stages_ - 1; s >= 0; --s) {
        const Tensor& y_in = stage_inputs[static_cast<std::size_t>(s)];
        Tensor gx(y_in.shape());
        gx.fill(0.0f);

        const int stride = 1 << s;
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n_; ++i) {
                const int j = i ^ stride;
                if (j < i) {
                    continue;
                }

                const int k = pair_index_for_stride(i, stride);

                const float xi = y_in({b, i});
                const float xj = y_in({b, j});

                const float w00 = stage_weights_[static_cast<std::size_t>(s)]({k, 0, 0});
                const float w01 = stage_weights_[static_cast<std::size_t>(s)]({k, 0, 1});
                const float w10 = stage_weights_[static_cast<std::size_t>(s)]({k, 1, 0});
                const float w11 = stage_weights_[static_cast<std::size_t>(s)]({k, 1, 1});

                const float gyi = gy({b, i});
                const float gyj = gy({b, j});

                gx({b, i}) += w00 * gyi + w10 * gyj;
                gx({b, j}) += w01 * gyi + w11 * gyj;

                grad_stages[static_cast<std::size_t>(s)]({k, 0, 0}) += gyi * xi;
                grad_stages[static_cast<std::size_t>(s)]({k, 0, 1}) += gyi * xj;
                grad_stages[static_cast<std::size_t>(s)]({k, 1, 0}) += gyj * xi;
                grad_stages[static_cast<std::size_t>(s)]({k, 1, 1}) += gyj * xj;
            }
        }
        gy = gx;
    }

    return BackwardGradients{gy, grad_stages};
}

ButterflyLinear ButterflyLinear::clone_weights() const {
    ButterflyLinear out(n_);
    for (int s = 0; s < num_stages_; ++s) {
        const Tensor& src = stage_weights_[static_cast<std::size_t>(s)];
        Tensor& dst = out.stage_weight(s);
        for (int k = 0; k < src.shape()[0]; ++k) {
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    dst({k, a, b}) = src({k, a, b});
                }
            }
        }
    }
    return out;
}
