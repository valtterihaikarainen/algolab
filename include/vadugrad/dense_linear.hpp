#pragma once

/**
 * @file dense_linear.hpp
 * @brief Fully connected (dense) linear layer: @f$y = x W + b@f$ with manual backprop.
 *
 * Shapes: @p x is @f$[B, d_{in}]@f$, weight @f$W@f$ is @f$[d_{in}, d_{out}]@f$, bias @f$b@f$ is
 * @f$[d_{out}]@f$, output @f$y@f$ is @f$[B, d_{out}]@f$ (row-major).
 */

#include <vadugrad/tensor.hpp>

/**
 * @brief Dense linear layer parameters and forward/backward without optimizer.
 */
class DenseLinear {
    Tensor weight_;  ///< Shape [in_features, out_features].
    Tensor bias_;    ///< Shape [out_features].
    Tensor gradient_;

public:
    /**
     * @brief Allocate weight and bias tensors (zero-initialized).
     */
    DenseLinear(int in_features, int out_features);

    /** @return Mutable weight tensor [in_features, out_features]. */
    Tensor& weight();
    /** @return Mutable bias tensor [out_features]. */
    Tensor& bias();
    /** @return Const weight. */
    const Tensor& weight() const;
    /** @return Const bias. */
    const Tensor& bias() const;

    int in_features() const;
    int out_features() const;

    /**
     * @brief Forward pass @f$y = x W + b@f$ (bias broadcast over batch).
     *
     * @param x Input of shape [batch, in_features].
     * @return Output of shape [batch, out_features].
     * @throws std::invalid_argument if @p x is not rank-2 or inner dim mismatches.
     */
    [[nodiscard]] Tensor forward(const Tensor& x) const;

    /**
     * @brief Gradients w.r.t. input, weight, and bias given upstream @f$\partial L/\partial y@f$.
     */
    struct BackwardGradients {
        Tensor grad_input;   ///< [batch, in_features]
        Tensor grad_weight;  ///< [in_features, out_features]
        Tensor grad_bias;    ///< [out_features]
    };

    /**
     * @param x Same tensor as used in @ref forward (shape [batch, in_features]).
     * @param grad_output @f$\partial L/\partial y@f$, shape [batch, out_features].
     */
    [[nodiscard]] BackwardGradients backward(const Tensor& x, const Tensor& grad_output) const;

};
