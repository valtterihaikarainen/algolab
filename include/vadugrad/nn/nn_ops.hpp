#pragma once

#include <vadugrad/tensor.hpp>

[[nodiscard]] Tensor softmax_last_dim(const Tensor& x);
[[nodiscard]] Tensor log_softmax_last_dim(const Tensor& x);
[[nodiscard]] float cross_entropy_mean(const Tensor& logits, const Tensor& target_ids,
                                       int ignore_index = -1);
[[nodiscard]] Tensor cross_entropy_grad_logits(const Tensor& logits, const Tensor& target_ids,
                                               int ignore_index = -1);
