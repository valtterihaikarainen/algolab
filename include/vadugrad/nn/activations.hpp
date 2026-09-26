#pragma once

#include <vadugrad/tensor.hpp>

[[nodiscard]] Tensor relu(const Tensor& x);
[[nodiscard]] Tensor relu_backward(const Tensor& x, const Tensor& grad_output);
[[nodiscard]] Tensor gelu(const Tensor& x);
[[nodiscard]] Tensor gelu_backward(const Tensor& x, const Tensor& grad_output);
