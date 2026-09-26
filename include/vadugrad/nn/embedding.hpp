#pragma once

#include <optional>
#include <vadugrad/tensor.hpp>

class Embedding {
    Tensor weight_;
    mutable std::optional<Tensor> cache_ids_;

public:
    Embedding(int vocab_size, int d_model);

    Tensor& weight();
    const Tensor& weight() const;

    [[nodiscard]] Tensor forward(const Tensor& token_ids) const;
    [[nodiscard]] Tensor backward(const Tensor& grad_output) const;
};
