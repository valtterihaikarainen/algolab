#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vadugrad/nn/activations.hpp>
#include <vadugrad/nn/nn_ops.hpp>

#include "grad_check.hpp"

namespace {

// Row [0, ln 2, ln 3] has exponentials [1, 2, 3], so its softmax is exactly [1/6, 2/6, 3/6].
Tensor hand_logits(float shift) {
    Tensor logits({2, 3});
    logits(0, 0) = shift;
    logits(0, 1) = shift + std::log(2.0f);
    logits(0, 2) = shift + std::log(3.0f);
    logits(1, 0) = shift;
    logits(1, 1) = shift;
    logits(1, 2) = shift;
    return logits;
}

Tensor labels_of(std::initializer_list<int> ids) {
    Tensor labels({static_cast<int>(ids.size()), 1});
    int row = 0;
    for (int id : ids) {
        labels(row++, 0) = static_cast<float>(id);
    }
    return labels;
}

}  // namespace

TEST(SoftmaxTest, MatchesHandComputation) {
    const Tensor probs = softmax_rows(hand_logits(0.0f));
    EXPECT_NEAR(probs(0, 0), 1.0f / 6.0f, 1e-6f);
    EXPECT_NEAR(probs(0, 1), 2.0f / 6.0f, 1e-6f);
    EXPECT_NEAR(probs(0, 2), 3.0f / 6.0f, 1e-6f);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(probs(1, c), 1.0f / 3.0f, 1e-6f);
    }
}

// Without subtracting the row maximum, exp(1000) overflows to infinity and the result is NaN.
// Softmax is shift-invariant, so the answer must equal the unshifted one.
TEST(SoftmaxTest, LargeLogitsAreStable) {
    const Tensor probs = softmax_rows(hand_logits(1000.0f));
    EXPECT_NEAR(probs(0, 0), 1.0f / 6.0f, 1e-4f);
    EXPECT_NEAR(probs(0, 2), 3.0f / 6.0f, 1e-4f);
    const Tensor log_probs = log_softmax_rows(hand_logits(-1000.0f));
    EXPECT_NEAR(log_probs(0, 2), std::log(0.5f), 1e-3f);
}

// Row 0 has p(class 2) = 1/2 and row 1 has p(class 1) = 1/3, so the mean loss is
// (ln 2 + ln 3) / 2.
TEST(CrossEntropyTest, MatchesHandComputation) {
    const float loss = cross_entropy_mean_rows(hand_logits(0.0f), labels_of({2, 1}));
    EXPECT_NEAR(loss, (std::log(2.0f) + std::log(3.0f)) / 2.0f, 1e-5f);
}

TEST(CrossEntropyTest, LargeLogitsGiveFiniteLoss) {
    const float loss = cross_entropy_mean_rows(hand_logits(1000.0f), labels_of({2, 1}));
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, (std::log(2.0f) + std::log(3.0f)) / 2.0f, 1e-3f);
}

// The analytic gradient (softmax - one_hot) / B must match finite differences of the loss on
// random logits of MNIST shape (10 classes).
TEST(CrossEntropyTest, GradientMatchesFiniteDifferences) {
    Tensor logits = grad_check::random_tensor({4, 10}, 51u, 2.0f);
    const Tensor labels = labels_of({0, 3, 9, 3});
    const Tensor analytic = cross_entropy_grad_logits_rows(logits, labels);
    auto loss = [&] { return static_cast<double>(cross_entropy_mean_rows(logits, labels)); };
    grad_check::expect_close(analytic, grad_check::numerical_gradient(logits, loss, 1e-3f), 1e-2f,
                             "logits");
}

TEST(CrossEntropyTest, InvalidLabelsThrow) {
    const Tensor logits = hand_logits(0.0f);
    EXPECT_THROW((void)cross_entropy_mean_rows(logits, labels_of({0, 3})), std::invalid_argument);
    EXPECT_THROW((void)cross_entropy_mean_rows(logits, labels_of({0, -1})), std::invalid_argument);
    EXPECT_THROW((void)cross_entropy_mean_rows(logits, labels_of({0})), std::invalid_argument);
    EXPECT_THROW((void)cross_entropy_grad_logits_rows(logits, labels_of({5, 0})), std::invalid_argument);
}

TEST(ReluTest, ForwardAndBackwardMatchHandComputation) {
    Tensor x({1, 4});
    x(0, 0) = -2.0f;
    x(0, 1) = -0.5f;
    x(0, 2) = 0.5f;
    x(0, 3) = 3.0f;
    const Tensor y = relu(x);
    EXPECT_FLOAT_EQ(y(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(y(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(y(0, 2), 0.5f);
    EXPECT_FLOAT_EQ(y(0, 3), 3.0f);

    Tensor upstream({1, 4});
    upstream.fill(2.0f);
    const Tensor grad = relu_backward(x, upstream);
    EXPECT_FLOAT_EQ(grad(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grad(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(grad(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(grad(0, 3), 2.0f);
}

// Inputs are kept at least 0.1 away from the kink at 0, where ReLU has no derivative.
TEST(ReluTest, GradientMatchesFiniteDifferences) {
    Tensor x = grad_check::random_tensor({3, 6}, 61u);
    for (int i = 0; i < x.numel(); ++i) {
        float& v = x.data()[i];
        v = v >= 0.0f ? v + 0.1f : v - 0.1f;
    }
    const Tensor upstream = grad_check::random_tensor({3, 6}, 62u);
    auto loss = [&] { return grad_check::weighted_sum(relu(x), upstream); };
    grad_check::expect_close(relu_backward(x, upstream), grad_check::numerical_gradient(x, loss), 1e-3f,
                             "relu");
}
