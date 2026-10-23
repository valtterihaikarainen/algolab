/**
 * @file nn_ops.cpp
 * @brief Softmax and cross-entropy. The public API is documented in nn_ops.hpp.
 */

#include <vadugrad/nn/nn_ops.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

/** Throw unless @p logits is [B, C]. */
void check_logits_rank2(const Tensor& logits, const char* name) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " expects rank-2 logits [B, C]");
    }
}

/** Throw unless @p logits is [B, C] and @p labels is [B, 1] with the same B. */
void check_logits_and_labels(const Tensor& logits, const Tensor& labels) {
    check_logits_rank2(logits, "cross-entropy");
    if (labels.ndim() != 2 || labels.shape()[1] != 1) {
        throw std::invalid_argument("rank-2 labels must be [B, 1]");
    }
    if (labels.shape()[0] != logits.shape()[0]) {
        throw std::invalid_argument("logits/labels batch mismatch");
    }
}

/** @return The class id stored in row @p row of @p labels, checked to be in [0, num_classes). */
int label_at(const Tensor& labels, int row, int num_classes) {
    const int label = static_cast<int>(labels(row, 0));
    if (label < 0 || label >= num_classes) {
        throw std::invalid_argument("cross-entropy: label out of range");
    }
    return label;
}

/** @return The largest score in row @p row of @p logits. */
float row_max(const Tensor& logits, int row) {
    float max_value = logits(row, 0);
    for (int c = 1; c < logits.shape()[1]; ++c) {
        max_value = std::max(max_value, logits(row, c));
    }
    return max_value;
}

}  // namespace

Tensor softmax_rows(const Tensor& logits) {
    check_logits_rank2(logits, "softmax_rows");
    const int batch = logits.shape()[0];
    const int num_classes = logits.shape()[1];
    Tensor probs(logits.shape());
    for (int b = 0; b < batch; ++b) {
        // softmax(z)_c = exp(z_c) / sum_k exp(z_k). Subtracting the row maximum first does not
        // change the result (it cancels in the fraction) but keeps exp() from overflowing:
        // the largest exponent becomes exp(0) = 1.
        const float max_value = row_max(logits, b);
        float denom = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            const float e = std::exp(logits(b, c) - max_value);
            probs(b, c) = e;
            denom += e;
        }
        for (int c = 0; c < num_classes; ++c) {
            probs(b, c) /= denom;
        }
    }
    return probs;
}

Tensor log_softmax_rows(const Tensor& logits) {
    check_logits_rank2(logits, "log_softmax_rows");
    const int batch = logits.shape()[0];
    const int num_classes = logits.shape()[1];
    Tensor log_probs(logits.shape());
    for (int b = 0; b < batch; ++b) {
        // log softmax(z)_c = z_c - log sum_k exp(z_k), and with m = max_k z_k the log-sum-exp
        // is computed as m + log sum_k exp(z_k - m). Taking log(softmax) directly would turn
        // a probability that rounds to 0 into -infinity.
        const float max_value = row_max(logits, b);
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(logits(b, c) - max_value);
        }
        const float log_denom = max_value + std::log(sum_exp);
        for (int c = 0; c < num_classes; ++c) {
            log_probs(b, c) = logits(b, c) - log_denom;
        }
    }
    return log_probs;
}

float cross_entropy_mean_rows(const Tensor& logits, const Tensor& labels) {
    check_logits_and_labels(logits, labels);
    const int batch = logits.shape()[0];
    const int num_classes = logits.shape()[1];
    // Loss of one example = -log(probability given to its correct class).
    const Tensor log_probs = log_softmax_rows(logits);
    float loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        loss -= log_probs(b, label_at(labels, b, num_classes));
    }
    return loss / static_cast<float>(batch);
}

Tensor cross_entropy_grad_logits_rows(const Tensor& logits, const Tensor& labels) {
    check_logits_and_labels(logits, labels);
    const int batch = logits.shape()[0];
    const int num_classes = logits.shape()[1];
    // Differentiating -log softmax(z)_y gives softmax(z) - one_hot(y): every wrong class is
    // pushed down by its probability, the correct class pushed up by 1 - p. Dividing by B
    // matches the mean taken in cross_entropy_mean_rows.
    Tensor grad = softmax_rows(logits);
    for (int b = 0; b < batch; ++b) {
        grad(b, label_at(labels, b, num_classes)) -= 1.0f;
    }
    const float inv_batch = 1.0f / static_cast<float>(batch);
    for (int i = 0; i < grad.numel(); ++i) {
        grad.data()[i] *= inv_batch;
    }
    return grad;
}
