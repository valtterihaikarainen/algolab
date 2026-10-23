/**
 * @file activations.cpp
 * @brief ReLU and its gradient. The public API is documented in activations.hpp.
 */

#include <vadugrad/nn/activations.hpp>

#include <stdexcept>

Tensor relu(const Tensor& x) {
    Tensor out(x.shape());
    for (int i = 0; i < x.numel(); ++i) {
        out.data()[i] = x.data()[i] > 0.0f ? x.data()[i] : 0.0f;
    }
    return out;
}

Tensor relu_backward(const Tensor& x, const Tensor& grad_output) {
    if (x.shape() != grad_output.shape()) {
        throw std::invalid_argument("relu_backward shape mismatch");
    }
    // The derivative of max(0, x) is 1 for x > 0 and 0 for x < 0. At exactly 0 it is
    // undefined; using 0 there is the usual convention.
    Tensor grad_input(x.shape());
    for (int i = 0; i < x.numel(); ++i) {
        grad_input.data()[i] = x.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
    }
    return grad_input;
}
