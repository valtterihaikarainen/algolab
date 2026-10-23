#pragma once

/**
 * @file activations.hpp
 * @brief ReLU activation and its gradient.
 */

#include <vadugrad/tensor.hpp>

/**
 * @brief Elementwise ReLU: max(0, x).
 * @param x Input of any shape.
 * @return Tensor of the same shape.
 */
[[nodiscard]] Tensor relu(const Tensor& x);

/**
 * @brief Backward pass of @ref relu.
 *
 * The gradient passes through where the forward input was positive and is zero elsewhere.
 * @param x The input that was given to @ref relu.
 * @param grad_output Gradient of the loss with respect to relu(x); same shape as @p x.
 * @return Gradient of the loss with respect to @p x.
 */
[[nodiscard]] Tensor relu_backward(const Tensor& x, const Tensor& grad_output);
