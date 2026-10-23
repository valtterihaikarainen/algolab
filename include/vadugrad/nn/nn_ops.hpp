#pragma once

/**
 * @file nn_ops.hpp
 * @brief Softmax and cross-entropy loss for a batch of classification scores.
 *
 * All functions take logits of shape [B, C]: one row of C unnormalized class scores per
 * example. Labels are stored as a [B, 1] tensor of integer class ids in [0, C).
 */

#include <vadugrad/tensor.hpp>

/**
 * @brief Row-wise softmax: turns each row of scores into probabilities that sum to 1.
 *
 * The row maximum is subtracted before exponentiating so large scores do not overflow.
 * @param logits Scores [B, C].
 * @return Probabilities [B, C].
 */
[[nodiscard]] Tensor softmax_rows(const Tensor& logits);

/**
 * @brief Row-wise logarithm of the softmax, computed stably with the log-sum-exp trick.
 * @param logits Scores [B, C].
 * @return Log-probabilities [B, C].
 */
[[nodiscard]] Tensor log_softmax_rows(const Tensor& logits);

/**
 * @brief Mean cross-entropy loss: the average of -log p(correct class) over the batch.
 * @param logits Scores [B, C].
 * @param labels Correct class ids [B, 1].
 * @return Scalar loss (0 means every correct class got probability 1).
 * @throws std::invalid_argument on shape mismatch or a label outside [0, C).
 */
[[nodiscard]] float cross_entropy_mean_rows(const Tensor& logits, const Tensor& labels);

/**
 * @brief Gradient of @ref cross_entropy_mean_rows with respect to the logits.
 *
 * Equals (softmax(logits) - one_hot(labels)) / B.
 * @param logits Scores [B, C].
 * @param labels Correct class ids [B, 1].
 * @return Gradient [B, C].
 */
[[nodiscard]] Tensor cross_entropy_grad_logits_rows(const Tensor& logits, const Tensor& labels);
