#pragma once

/**
 * @file mlp.hpp
 * @brief The classifier used in the experiment: a small multilayer perceptron whose hidden
 *        layer is absent, dense, or butterfly-structured.
 *
 * Architecture (all variants share the input and output layers):
 *
 *     input --Dense--> ReLU --[hidden layer]--> ReLU --Dense--> logits
 *
 * - None:      the hidden layer and its ReLU are skipped.
 * - Dense:     the hidden layer is a full hidden_dim × hidden_dim @ref DenseLinear.
 * - Butterfly: the hidden layer is a @ref ButterflyLinear of width hidden_dim.
 */

#include <optional>
#include <vector>

#include <vadugrad/butterfly_linear.hpp>
#include <vadugrad/dense_linear.hpp>
#include <vadugrad/optim/adam.hpp>
#include <vadugrad/tensor.hpp>

/** Which hidden layer the network uses. */
enum class HiddenVariant { None, Dense, Butterfly };

class Mlp {
    HiddenVariant variant_;
    DenseLinear input_layer_;
    std::optional<DenseLinear> dense_hidden_;
    std::optional<ButterflyLinear> butterfly_hidden_;
    DenseLinear output_layer_;

    // Activations from the latest forward call, needed by backward.
    Tensor input_;
    Tensor input_layer_out_;   ///< Before the first ReLU.
    Tensor hidden_in_;         ///< After the first ReLU.
    Tensor hidden_layer_out_;  ///< Before the second ReLU (unused for HiddenVariant::None).
    Tensor output_layer_in_;

    std::vector<Tensor> gradients_;  ///< Same order as @ref parameters.

public:
    /**
     * @param input_dim Number of input features (784 for MNIST).
     * @param hidden_dim Width of the hidden representation; must be a power of two for
     *        HiddenVariant::Butterfly.
     * @param num_classes Number of output classes.
     * @param variant Hidden-layer type.
     * @throws std::invalid_argument for non-positive sizes or a butterfly width that is not a
     *         power of two.
     */
    Mlp(int input_dim, int hidden_dim, int num_classes, HiddenVariant variant);

    /**
     * @brief Draw random initial weights; all biases start at zero.
     *
     * Layers followed by a ReLU use He initialization (std = sqrt(2 / fan_in)), the output
     * layer uses std = sqrt(1 / fan_in). Butterfly blocks have fan-in 2 and use std = 1/sqrt(2),
     * so every stage keeps the variance of the signal unchanged.
     * @param seed Non-zero seed; the same seed always gives the same network.
     */
    void initialize(unsigned int seed);

    /**
     * @brief Compute class scores and cache activations for @ref backward.
     * @param x Inputs [batch, input_dim].
     * @return Logits [batch, num_classes].
     */
    [[nodiscard]] Tensor forward(const Tensor& x);

    /**
     * @brief Backpropagate through the whole network for the latest @ref forward call and
     *        store the parameter gradients (see @ref gradients).
     * @param grad_logits Gradient of the loss with respect to the logits, [batch, num_classes].
     */
    void backward(const Tensor& grad_logits);

    /** @brief Update every parameter with its stored gradient. */
    void apply_gradients(Adam& optimizer);

    /** @return Pointers to every learnable tensor, input layer first. */
    std::vector<Tensor*> parameters();

    /** @return Gradients from the latest @ref backward, in the order of @ref parameters. */
    const std::vector<Tensor>& gradients() const;

    /** @return Number of learnable values in the hidden layer (0 for HiddenVariant::None). */
    int hidden_parameter_count() const;

    /** @return Number of learnable values in the whole network. */
    int parameter_count() const;

    HiddenVariant variant() const;
};
