#pragma once

#include <optional>
#include <vadugrad/core/tensor_ops.hpp>
#include <vadugrad/dense_linear.hpp>

class FeedForward {
    DenseLinear fc1_;
    DenseLinear fc2_;
    mutable std::optional<Tensor> cache_input_;
    mutable std::optional<Tensor> cache_preact_;

public:
    FeedForward(int d_model, int d_ff);

    DenseLinear& fc1();
    DenseLinear& fc2();
    const DenseLinear& fc1() const;
    const DenseLinear& fc2() const;

    [[nodiscard]] Tensor forward(const Tensor& x) const;

    struct BackwardGradients {
        Tensor grad_input;
        Tensor grad_fc1_weight;
        Tensor grad_fc1_bias;
        Tensor grad_fc2_weight;
        Tensor grad_fc2_bias;
    };

    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output) const;
};
