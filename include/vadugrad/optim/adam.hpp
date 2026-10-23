#pragma once

/**
 * @file adam.hpp
 * @brief Adam optimizer (Kingma & Ba, 2015).
 */

#include <unordered_map>

#include <vadugrad/tensor.hpp>

/**
 * @brief Adam optimizer with separate moment estimates for every parameter tensor.
 *
 * For each parameter Adam keeps a running mean of gradients (m) and of squared gradients
 * (v), and moves the parameter by lr * m_hat / (sqrt(v_hat) + eps). The hats denote bias
 * correction by 1 - beta^t, where t counts the updates of that particular parameter.
 *
 * Parameters are identified by the address of their storage, so a parameter tensor must not
 * be reallocated between calls.
 */
class Adam {
    struct ParamState {
        Tensor m;  ///< Running mean of gradients.
        Tensor v;  ///< Running mean of squared gradients.
        int step;  ///< Number of updates applied to this parameter so far.
    };

    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    std::unordered_map<const float*, ParamState> states_;

public:
    /**
     * @param lr Learning rate (step size).
     * @param beta1 Decay rate of the gradient mean.
     * @param beta2 Decay rate of the squared-gradient mean.
     * @param eps Small constant that prevents division by zero.
     */
    explicit Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    /**
     * @brief Apply one Adam update to @p param in place.
     * @param param Parameter tensor to update.
     * @param grad Gradient of the loss with respect to @p param; same shape.
     * @throws std::invalid_argument if the shapes differ.
     */
    void step(Tensor& param, const Tensor& grad);
};
