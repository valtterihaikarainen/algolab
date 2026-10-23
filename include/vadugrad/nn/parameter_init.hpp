#pragma once

/**
 * @file parameter_init.hpp
 * @brief Reproducible random initialization of parameter tensors.
 */

#include <vadugrad/tensor.hpp>

/**
 * @brief Fill @p tensor with samples from a normal distribution N(0, stddev²).
 *
 * Uses a xorshift32 generator and the Box–Muller transform, so the same @p seed always
 * gives the same values on every platform.
 *
 * @param tensor Tensor to overwrite.
 * @param stddev Standard deviation; must be positive.
 * @param seed Generator state; must be non-zero. Advanced in place so consecutive calls
 *             produce different values.
 * @throws std::invalid_argument if @p stddev <= 0 or @p seed == 0.
 */
void initialize_tensor_normal(Tensor& tensor, float stddev, unsigned int& seed);
