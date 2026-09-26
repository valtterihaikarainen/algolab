#pragma once

#include <optional>
#include <vadugrad/tensor.hpp>

class LayerNorm {
    Tensor gamma_;
    Tensor beta_;
    float eps_;
    mutable std::optional<Tensor> cache_input_;
    mutable std::optional<Tensor> cache_mean_;
    mutable std::optional<Tensor> cache_inv_std_;

public:
    explicit LayerNorm(int d_model, float eps = 1e-5f);

    Tensor& gamma();
    Tensor& beta();
    const Tensor& gamma() const;
    const Tensor& beta() const;

    [[nodiscard]] Tensor forward(const Tensor& x) const;

    struct BackwardGradients {
        Tensor grad_input;
        Tensor grad_gamma;
        Tensor grad_beta;
    };

    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output) const;
};
