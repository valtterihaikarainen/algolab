#pragma once

/**
 * @file butterfly_linear.hpp
 * @brief Butterfly-structured linear layer y = B x + b with manual backpropagation.
 *
 * The n × n matrix B is never stored. It is the product of log2(n) sparse stages. In
 * stage s, every position i is paired with position i XOR 2^s, and each pair is mixed by its
 * own learnable 2×2 block:
 *
 *     out[i] = w00 * in[i] + w01 * in[j]
 *     out[j] = w10 * in[i] + w11 * in[j]      (i < j = i XOR 2^s)
 *
 * This is the wiring of the Fast Fourier Transform with learnable blocks in place of the
 * fixed FFT coefficients. Cost per input vector: O(n log n) time and 2 n log2(n) weights,
 * compared with O(n²) for a dense layer.
 *
 * Shapes: input and output are [batch, n] (row-major).
 */

#include <optional>
#include <vector>

#include <vadugrad/tensor.hpp>

class ButterflyLinear {
    int n_;
    int num_stages_;
    std::vector<Tensor> stage_weights_;  ///< Per stage: [n/2, 2, 2], one 2×2 block per pair.
    Tensor bias_;                        ///< [n], added after the last stage.

    /** Input of every stage from the latest forward call; needed by backward. */
    mutable std::optional<std::vector<Tensor>> cache_stage_inputs_;

public:
    /**
     * @brief Create a layer with all weights and the bias set to zero.
     * @param n Vector width; must be a power of two and at least 2.
     * @throws std::invalid_argument otherwise.
     */
    explicit ButterflyLinear(int n);

    /** @return Vector width n. */
    int dim() const;

    /** @return Number of stages, log2(n). */
    int num_stages() const;

    /** @return Total number of learnable values: 2 n log2(n) block weights plus n biases. */
    int num_parameters() const;

    /**
     * @param s Stage index in [0, num_stages()).
     * @return Mutable block weights of stage @p s, shape [n/2, 2, 2].
     */
    Tensor& stage_weight(int s);
    const Tensor& stage_weight(int s) const;

    /** @return Mutable bias, shape [n]. */
    Tensor& bias();
    const Tensor& bias() const;

    /**
     * @brief Forward pass y = B x + b. Caches every stage input for @ref backward.
     * @param x Input [batch, n].
     * @return Output [batch, n].
     * @throws std::invalid_argument if @p x is not [batch, n].
     */
    [[nodiscard]] Tensor forward(const Tensor& x) const;

    /** Gradients of the loss with respect to the input and every parameter. */
    struct BackwardGradients {
        Tensor grad_input;                ///< [batch, n]
        std::vector<Tensor> grad_stages;  ///< One [n/2, 2, 2] tensor per stage
        Tensor grad_bias;                 ///< [n]
    };

    /**
     * @brief Backward pass for the latest @ref forward call.
     *
     * Walks the stages in reverse. For each pair the 2×2 block is transposed to route the
     * gradient back to the inputs, and outer products of upstream gradient and cached input
     * give the block gradients.
     * @param grad_output Gradient of the loss with respect to the output, [batch, n].
     * @return Gradients for the input, the stage weights and the bias.
     * @throws std::invalid_argument if called before @ref forward or on a shape mismatch.
     */
    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output) const;
};
