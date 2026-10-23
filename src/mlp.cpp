/**
 * @file mlp.cpp
 * @brief Implementation of @ref Mlp: wiring the layers together and backpropagating through
 *        all of them. The public API is documented in mlp.hpp.
 */

#include <vadugrad/nn/mlp.hpp>

#include <cmath>
#include <stdexcept>

#include <vadugrad/nn/activations.hpp>
#include <vadugrad/nn/parameter_init.hpp>

Mlp::Mlp(int input_dim, int hidden_dim, int num_classes, HiddenVariant variant)
    : variant_(variant),
      input_layer_(input_dim, hidden_dim),
      output_layer_(hidden_dim, num_classes),
      input_({1}),
      input_layer_out_({1}),
      hidden_in_({1}),
      hidden_layer_out_({1}),
      output_layer_in_({1}) {
    if (variant == HiddenVariant::Dense) {
        dense_hidden_.emplace(hidden_dim, hidden_dim);
    } else if (variant == HiddenVariant::Butterfly) {
        butterfly_hidden_.emplace(hidden_dim);
    }
}

void Mlp::initialize(unsigned int seed) {
    // The standard deviations are chosen so that the signal keeps roughly the same size from
    // layer to layer. Too small and it fades to zero after a few layers, too large and it
    // explodes. A ReLU zeroes about half of its inputs, which He initialization compensates
    // with the extra factor 2.
    const auto he_std = [](const DenseLinear& layer) {
        return std::sqrt(2.0f / static_cast<float>(layer.in_features()));
    };
    initialize_tensor_normal(input_layer_.weight(), he_std(input_layer_), seed);
    input_layer_.bias().fill(0.0f);
    if (dense_hidden_) {
        initialize_tensor_normal(dense_hidden_->weight(), he_std(*dense_hidden_), seed);
        dense_hidden_->bias().fill(0.0f);
    }
    if (butterfly_hidden_) {
        // Each output of a 2×2 block sums two inputs, so std 1/sqrt(2) keeps the variance of
        // every stage at 1. With the old fixed std of 0.02 the signal shrank about 35-fold per
        // stage and was gone after nine stages.
        for (int s = 0; s < butterfly_hidden_->num_stages(); ++s) {
            initialize_tensor_normal(butterfly_hidden_->stage_weight(s), 1.0f / std::sqrt(2.0f), seed);
        }
        butterfly_hidden_->bias().fill(0.0f);
    }
    const float output_std = std::sqrt(1.0f / static_cast<float>(output_layer_.in_features()));
    initialize_tensor_normal(output_layer_.weight(), output_std, seed);
    output_layer_.bias().fill(0.0f);
}

Tensor Mlp::forward(const Tensor& x) {
    // Every intermediate result is kept in a member: the backward pass needs each layer's
    // input (for the weight gradients) and each ReLU's input (to know where it was active).
    input_ = x;
    input_layer_out_ = input_layer_.forward(x);
    hidden_in_ = relu(input_layer_out_);

    if (dense_hidden_) {
        hidden_layer_out_ = dense_hidden_->forward(hidden_in_);
        output_layer_in_ = relu(hidden_layer_out_);
    } else if (butterfly_hidden_) {
        hidden_layer_out_ = butterfly_hidden_->forward(hidden_in_);
        output_layer_in_ = relu(hidden_layer_out_);
    } else {
        output_layer_in_ = hidden_in_;
    }
    return output_layer_.forward(output_layer_in_);
}

void Mlp::backward(const Tensor& grad_logits) {
    // Backpropagation runs the forward pass in reverse (chain rule). Each layer turns the
    // gradient with respect to its output into gradients for its own parameters and for its
    // input, and that input gradient is the output gradient of the layer before it.
    auto output_grads = output_layer_.backward(output_layer_in_, grad_logits);

    std::vector<Tensor> hidden_grads;
    Tensor grad_hidden_in = std::move(output_grads.grad_input);
    if (dense_hidden_) {
        // Second ReLU first: it only lets gradient through where its input was positive.
        const Tensor grad_pre = relu_backward(hidden_layer_out_, grad_hidden_in);
        auto grads = dense_hidden_->backward(hidden_in_, grad_pre);
        grad_hidden_in = std::move(grads.grad_input);
        hidden_grads.push_back(std::move(grads.grad_weight));
        hidden_grads.push_back(std::move(grads.grad_bias));
    } else if (butterfly_hidden_) {
        const Tensor grad_pre = relu_backward(hidden_layer_out_, grad_hidden_in);
        auto grads = butterfly_hidden_->backward(grad_pre);
        grad_hidden_in = std::move(grads.grad_input);
        for (auto& g : grads.grad_stages) {
            hidden_grads.push_back(std::move(g));
        }
        hidden_grads.push_back(std::move(grads.grad_bias));
    }
    // For HiddenVariant::None there is no hidden layer, so the output layer's input gradient
    // goes straight to the first ReLU.

    const Tensor grad_input_pre = relu_backward(input_layer_out_, grad_hidden_in);
    auto input_grads = input_layer_.backward(input_, grad_input_pre);

    // Store in the same order as parameters(), so apply_gradients can pair them by index.
    gradients_.clear();
    gradients_.push_back(std::move(input_grads.grad_weight));
    gradients_.push_back(std::move(input_grads.grad_bias));
    for (auto& g : hidden_grads) {
        gradients_.push_back(std::move(g));
    }
    gradients_.push_back(std::move(output_grads.grad_weight));
    gradients_.push_back(std::move(output_grads.grad_bias));
}

void Mlp::apply_gradients(Adam& optimizer) {
    const std::vector<Tensor*> params = parameters();
    if (gradients_.size() != params.size()) {
        throw std::logic_error("Mlp::apply_gradients called before backward");
    }
    for (std::size_t i = 0; i < params.size(); ++i) {
        optimizer.step(*params[i], gradients_[i]);
    }
}

std::vector<Tensor*> Mlp::parameters() {
    std::vector<Tensor*> params{&input_layer_.weight(), &input_layer_.bias()};
    if (dense_hidden_) {
        params.push_back(&dense_hidden_->weight());
        params.push_back(&dense_hidden_->bias());
    }
    if (butterfly_hidden_) {
        for (int s = 0; s < butterfly_hidden_->num_stages(); ++s) {
            params.push_back(&butterfly_hidden_->stage_weight(s));
        }
        params.push_back(&butterfly_hidden_->bias());
    }
    params.push_back(&output_layer_.weight());
    params.push_back(&output_layer_.bias());
    return params;
}

const std::vector<Tensor>& Mlp::gradients() const { return gradients_; }

int Mlp::hidden_parameter_count() const {
    if (dense_hidden_) {
        return dense_hidden_->weight().numel() + dense_hidden_->bias().numel();
    }
    if (butterfly_hidden_) {
        return butterfly_hidden_->num_parameters();
    }
    return 0;
}

int Mlp::parameter_count() const {
    return input_layer_.weight().numel() + input_layer_.bias().numel() + hidden_parameter_count() +
           output_layer_.weight().numel() + output_layer_.bias().numel();
}

HiddenVariant Mlp::variant() const { return variant_; }
