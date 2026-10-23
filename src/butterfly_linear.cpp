/**
 * @file butterfly_linear.cpp
 * @brief Implementation of @ref ButterflyLinear.
 */

#include <vadugrad/butterfly_linear.hpp>

#include <stdexcept>
#include <string>

namespace {

bool is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

int log2_int(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        ++log;
    }
    return log;
}

/**
 * Block index of the pair whose lower element is @p i in stage @p stage.
 *
 * The lower element has bit @p stage equal to 0. Deleting that bit from @p i numbers the
 * n/2 pairs of a stage as 0, 1, ..., n/2 - 1 without gaps.
 */
int pair_index(int i, int stage) {
    const int low_bits = i & ((1 << stage) - 1);
    const int high_bits = i >> (stage + 1);
    return (high_bits << stage) | low_bits;
}

void validate_input(const Tensor& x, int n, const char* name) {
    if (x.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be rank-2 [batch, n]");
    }
    if (x.shape()[1] != n) {
        throw std::invalid_argument(std::string(name) + " inner dimension must equal n");
    }
}

}  // namespace

ButterflyLinear::ButterflyLinear(int n)
    : n_(n), num_stages_(log2_int(n)), bias_(std::vector<int>{n > 0 ? n : 1}) {
    if (n < 2 || !is_power_of_two(n)) {
        throw std::invalid_argument("ButterflyLinear: n must be a power of two and >= 2");
    }
    stage_weights_.reserve(static_cast<std::size_t>(num_stages_));
    for (int s = 0; s < num_stages_; ++s) {
        stage_weights_.emplace_back(std::vector<int>{n / 2, 2, 2});
    }
}

int ButterflyLinear::dim() const { return n_; }
int ButterflyLinear::num_stages() const { return num_stages_; }
int ButterflyLinear::num_parameters() const { return num_stages_ * (n_ / 2) * 4 + n_; }

Tensor& ButterflyLinear::stage_weight(int s) { return stage_weights_.at(static_cast<std::size_t>(s)); }
const Tensor& ButterflyLinear::stage_weight(int s) const {
    return stage_weights_.at(static_cast<std::size_t>(s));
}

Tensor& ButterflyLinear::bias() { return bias_; }
const Tensor& ButterflyLinear::bias() const { return bias_; }

Tensor ButterflyLinear::forward(const Tensor& x) const {
    validate_input(x, n_, "ButterflyLinear::forward x");
    const int batch = x.shape()[0];

    // The input of every stage is saved: the gradient of a block's weights depends on the
    // values that entered that block, so backward needs them all. This costs B·n·log2(n)
    // floats of memory.
    std::vector<Tensor> stage_inputs;
    stage_inputs.reserve(static_cast<std::size_t>(num_stages_));

    Tensor current = x;
    for (int s = 0; s < num_stages_; ++s) {
        // Stage s pairs positions that differ only in bit s: (0,1),(2,3),... for s = 0,
        // (0,2),(1,3),... for s = 1, and so on. Each pair is processed once, from its lower
        // element, and mixed by its own 2×2 block k.
        const Tensor& w = stage_weights_[static_cast<std::size_t>(s)];
        const int stride = 1 << s;
        Tensor next(current.shape());
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n_; ++i) {
                if (i & stride) {
                    continue;  // i is the upper element; its pair was handled at i - stride
                }
                const int j = i + stride;
                const int k = pair_index(i, s);
                const float xi = current(b, i);
                const float xj = current(b, j);
                next(b, i) = w(k, 0, 0) * xi + w(k, 0, 1) * xj;
                next(b, j) = w(k, 1, 0) * xi + w(k, 1, 1) * xj;
            }
        }
        stage_inputs.push_back(std::move(current));
        current = std::move(next);
    }

    // One bias per output position, added after the last stage (y = B x + b).
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n_; ++i) {
            current(b, i) += bias_.data()[i];
        }
    }

    cache_stage_inputs_ = std::move(stage_inputs);
    return current;
}

ButterflyLinear::BackwardGradients ButterflyLinear::backward(const Tensor& grad_output) const {
    if (!cache_stage_inputs_) {
        throw std::invalid_argument("ButterflyLinear::backward called before forward");
    }
    validate_input(grad_output, n_, "ButterflyLinear::backward grad_output");
    const auto& stage_inputs = *cache_stage_inputs_;
    const int batch = grad_output.shape()[0];
    if (stage_inputs.front().shape()[0] != batch) {
        throw std::invalid_argument("ButterflyLinear::backward: batch size differs from forward");
    }

    // The bias is added to every row, so its gradient is the upstream gradient summed over
    // the batch.
    Tensor grad_bias = sum(grad_output, 0, false);

    std::vector<Tensor> grad_stages;
    grad_stages.reserve(stage_weights_.size());
    for (const auto& w : stage_weights_) {
        grad_stages.emplace_back(w.shape());
    }

    // The last stage produced the output, so it is undone first. `grad` always holds the
    // gradient with respect to the output of stage s.
    Tensor grad = grad_output;
    for (int s = num_stages_ - 1; s >= 0; --s) {
        const Tensor& w = stage_weights_[static_cast<std::size_t>(s)];
        const Tensor& input = stage_inputs[static_cast<std::size_t>(s)];
        Tensor& grad_w = grad_stages[static_cast<std::size_t>(s)];
        const int stride = 1 << s;
        Tensor grad_input(input.shape());
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n_; ++i) {
                if (i & stride) {
                    continue;
                }
                const int j = i + stride;
                const int k = pair_index(i, s);
                const float xi = input(b, i);
                const float xj = input(b, j);
                const float gi = grad(b, i);
                const float gj = grad(b, j);

                // The forward pass computed [out_i, out_j] = W [x_i, x_j], so the gradient
                // flows back through the transpose: [g_xi, g_xj] = W^T [g_i, g_j]. Note w10,
                // not w01, in the first line.
                grad_input(b, i) = w(k, 0, 0) * gi + w(k, 1, 0) * gj;
                grad_input(b, j) = w(k, 0, 1) * gi + w(k, 1, 1) * gj;

                // out_i = w00 x_i + w01 x_j, so dL/dw00 = g_i x_i, dL/dw01 = g_i x_j, and
                // likewise for out_j. The same block is used by every row of the batch, so
                // the contributions are summed (+=).
                grad_w(k, 0, 0) += gi * xi;
                grad_w(k, 0, 1) += gi * xj;
                grad_w(k, 1, 0) += gj * xi;
                grad_w(k, 1, 1) += gj * xj;
            }
        }
        grad = std::move(grad_input);
    }

    return BackwardGradients{std::move(grad), std::move(grad_stages), std::move(grad_bias)};
}
