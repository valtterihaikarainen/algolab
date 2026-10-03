#pragma once

#include <vadugrad/tensor.hpp>

[[nodiscard]] Tensor softmax_last_dim(const Tensor& x);
[[nodiscard]] Tensor log_softmax_last_dim(const Tensor& x);
[[nodiscard]] float cross_entropy_mean(const Tensor& logits, const Tensor& target_ids,
                                       int ignore_index = -1);
[[nodiscard]] Tensor cross_entropy_grad_logits(const Tensor& logits, const Tensor& target_ids,
                                               int ignore_index = -1);

/** @brief Softmax over the last dimension for rank-2 logits [B, C]. */
[[nodiscard]] Tensor softmax_rows(const Tensor& logits);

/** @brief Log-softmax over the last dimension for rank-2 logits [B, C]. */
[[nodiscard]] Tensor log_softmax_rows(const Tensor& logits);

/** @brief Mean cross-entropy for rank-2 logits [B, C] and labels [B, 1]. */
[[nodiscard]] float cross_entropy_mean_rows(const Tensor& logits, const Tensor& labels);

/** @brief Gradient of @ref cross_entropy_mean_rows w.r.t. logits [B, C]. */
[[nodiscard]] Tensor cross_entropy_grad_logits_rows(const Tensor& logits, const Tensor& labels);
