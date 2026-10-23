#pragma once

/**
 * @file grad_check.hpp
 * @brief Finite-difference gradient checking shared by the layer tests.
 *
 * Backpropagation bugs (a missing transpose, a wrong index) usually still produce tensors of
 * the right shape, so shape checks cannot catch them. Instead, each analytic gradient is
 * compared against the central difference (L(p + h) - L(p - h)) / 2h, computed by nudging
 * one parameter element at a time.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vadugrad/nn/parameter_init.hpp>
#include <vadugrad/tensor.hpp>

namespace grad_check {

/** @return A tensor of @p shape filled with N(0, stddev²) samples from @p seed. */
inline Tensor random_tensor(const std::vector<int>& shape, unsigned int seed, float stddev = 1.0f) {
    Tensor t(shape);
    initialize_tensor_normal(t, stddev, seed);
    return t;
}

/**
 * @brief Scalar test loss L = sum_i weights_i * output_i.
 *
 * Its gradient with respect to @p output is exactly @p weights, so @p weights can be fed to
 * a layer's backward pass as the upstream gradient.
 */
inline double weighted_sum(const Tensor& output, const Tensor& weights) {
    double total = 0.0;
    for (int i = 0; i < output.numel(); ++i) {
        total += static_cast<double>(output.data()[i]) * static_cast<double>(weights.data()[i]);
    }
    return total;
}

/**
 * @brief Central-difference gradient of @p loss with respect to every element of @p param.
 * @param param Tensor that @p loss reads; each element is perturbed and then restored.
 * @param loss Callable returning the scalar loss for the current contents of @p param.
 * @param step Perturbation size h.
 */
template <typename LossFn>
Tensor numerical_gradient(Tensor& param, LossFn loss, float step = 1e-2f) {
    Tensor grad(param.shape());
    for (int i = 0; i < param.numel(); ++i) {
        const float saved = param.data()[i];
        param.data()[i] = saved + step;
        const double loss_plus = loss();
        param.data()[i] = saved - step;
        const double loss_minus = loss();
        param.data()[i] = saved;
        grad.data()[i] = static_cast<float>((loss_plus - loss_minus) / (2.0 * step));
    }
    return grad;
}

/**
 * @brief Expect every element to satisfy |analytic - numeric| <= tol * max(1, |numeric|).
 * @param label Name shown in failure messages.
 */
inline void expect_close(const Tensor& analytic, const Tensor& numeric, float tol,
                         const std::string& label) {
    ASSERT_EQ(analytic.shape(), numeric.shape()) << label;
    for (int i = 0; i < analytic.numel(); ++i) {
        const float a = analytic.data()[i];
        const float n = numeric.data()[i];
        const float scale = std::fmax(1.0f, std::fabs(n));
        EXPECT_LE(std::fabs(a - n), tol * scale) << label << " element " << i << ": analytic " << a
                                                 << " vs numeric " << n;
    }
}

}  // namespace grad_check
