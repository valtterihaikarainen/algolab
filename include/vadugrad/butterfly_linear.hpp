#pragma once

/**
 * @file butterfly_linear.hpp
 * @brief Butterfly-structured linear map @f$y = B x@f$ with manual backprop.
 *
 * This is a minimal, review-friendly butterfly factorization for @f$n@f$ a power of two,
 * block size @f$b=2@f$, and @f$L=\log_2 n@f$ stages. Each stage applies independent @f$2\times 2@f$
 * transforms on a fixed FFT-style pairing pattern.
 *
 * Shapes: @p x is @f$[B, n]@f$, output @f$y@f$ is @f$[B, n]@f$ (row-major).
 */

#include <optional>
#include <vector>

#include <vadugrad/tensor.hpp>

class ButterflyLinear {
    int n_;
    int num_stages_;
    std::vector<Tensor> stage_weights_;  ///< Per stage: [n/2, 2, 2] learnable 2x2 blocks.

    mutable std::optional<std::vector<Tensor>> cache_stage_inputs_;

public:
    /**
     * @param n Vector dimension; must be a power of two and >= 2.
     */
    explicit ButterflyLinear(int n);

    int dim() const;
    int num_stages() const;

    /** @return Mutable weight tensor for stage @p s, shape [n/2, 2, 2]. */
    Tensor& stage_weight(int s);
    const Tensor& stage_weight(int s) const;

    /**
     * @brief Forward pass @f$y = B x@f$.
     * @param x Input [batch, n].
     */
    [[nodiscard]] Tensor forward(const Tensor& x) const;

    struct BackwardGradients {
        Tensor grad_input;                 ///< [batch, n]
        std::vector<Tensor> grad_stages;  ///< Same shapes as stage_weights_
    };

    /**
     * @brief Backward for the latest @ref forward call.
     * @param grad_output Upstream gradient [batch, n].
     */
    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output) const;

    /**
     * @brief Deep copy with identical weights (fresh caches).
     *
     * Useful for analysis helpers that need multiple independent forward passes.
     */
    [[nodiscard]] ButterflyLinear clone_weights() const;
};
