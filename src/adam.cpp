/**
 * @file adam.cpp
 * @brief Implementation of the @ref Adam optimizer. The public API is documented in adam.hpp.
 */

#include <vadugrad/optim/adam.hpp>

#include <cmath>
#include <stdexcept>

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

void Adam::step(Tensor& param, const Tensor& grad) {
    if (param.shape() != grad.shape()) {
        throw std::invalid_argument("Adam::step param and grad shape mismatch");
    }
    // First update of this parameter: start both running averages at zero.
    auto it = states_.find(param.data());
    if (it == states_.end()) {
        it = states_.emplace(param.data(), ParamState{Tensor(param.shape()), Tensor(param.shape()), 0})
                 .first;
    }
    ParamState& state = it->second;
    ++state.step;
    // Because m and v start at zero, their early values are biased towards zero (after the
    // first step m = (1 - beta1) g, only a tenth of g). Dividing by 1 - beta^t removes this
    // bias; the correction approaches 1 as t grows.
    const float b1_correction = 1.0f - std::pow(beta1_, static_cast<float>(state.step));
    const float b2_correction = 1.0f - std::pow(beta2_, static_cast<float>(state.step));

    float* p = param.data();
    const float* g = grad.data();
    float* m = state.m.data();
    float* v = state.v.data();
    for (int i = 0; i < param.numel(); ++i) {
        // m: exponentially weighted average of the gradient (the direction to move).
        // v: the same for the squared gradient (how large the gradient typically is).
        // The step m_hat / sqrt(v_hat) is about ±1 in size, so every parameter moves by
        // roughly lr regardless of how large or small its raw gradient is.
        m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
        v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
        const float m_hat = m[i] / b1_correction;
        const float v_hat = v[i] / b2_correction;
        p[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
}
