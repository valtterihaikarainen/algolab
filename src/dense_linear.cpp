/**
 * @file dense_linear.cpp
 * @brief Implementation of @ref DenseLinear.
 */

#include <vadugrad/dense_linear.hpp>
#include <stdexcept>
#include <vector>

namespace {

std::vector<int> checked_linear_weight_shape(int in_features, int out_features) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("DenseLinear: feature dimensions must be positive");
    }
    return {in_features, out_features};
}

}  // namespace

DenseLinear::DenseLinear(int in_features, int out_features)
    : weight_(checked_linear_weight_shape(in_features, out_features)),
      bias_(std::vector<int>{out_features}) {}

Tensor& DenseLinear::weight() {
    return weight_;
}

Tensor& DenseLinear::bias() {
    return bias_;
}

const Tensor& DenseLinear::weight() const {
    return weight_;
}

const Tensor& DenseLinear::bias() const {
    return bias_;
}

int DenseLinear::in_features() const {
    return weight_.shape()[0];
}

int DenseLinear::out_features() const {
    return weight_.shape()[1];
}

Tensor DenseLinear::forward(const Tensor& x) const {
    if (x.ndim() != 2) {
        throw std::invalid_argument("DenseLinear::forward expects rank-2 input [batch, in_features]");
    }
    const auto xs = x.shape();
    if (xs[1] != in_features()) {
        throw std::invalid_argument("DenseLinear::forward: input last dim must match in_features");
    }

    Tensor y = matmul(x, weight_);
    const int batch = y.shape()[0];
    const int out_dim = y.shape()[1];
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_dim; ++o) {
            y({b, o}) += bias_({o});
        }
    }
    return y;
}

DenseLinear::BackwardGradients DenseLinear::backward(const Tensor& x, const Tensor& grad_output) const {
    if (x.ndim() != 2 || grad_output.ndim() != 2) {
        throw std::invalid_argument("DenseLinear::backward expects rank-2 x and grad_output");
    }
    const auto xs = x.shape();
    const auto gs = grad_output.shape();
    if (xs[1] != in_features()) {
        throw std::invalid_argument("DenseLinear::backward: x inner dim mismatch");
    }
    if (gs[0] != xs[0] || gs[1] != out_features()) {
        throw std::invalid_argument("DenseLinear::backward: grad_output shape must be [batch, out_features]");
    }

    const Tensor w_t = transpose2d(weight_);
    const Tensor x_t = transpose2d(x);
    return BackwardGradients{matmul(grad_output, w_t), matmul(x_t, grad_output),
                             sum(grad_output, 0, false)};
}
