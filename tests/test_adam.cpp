#include <gtest/gtest.h>

#include <stdexcept>
#include <vadugrad/optim/adam.hpp>

// With zero initial moments, bias correction makes m_hat = g and v_hat = g² on the first
// update, so each element moves by exactly lr * g / (|g| + eps), i.e. about lr * sign(g).
TEST(AdamTest, FirstUpdateMatchesHandComputation) {
    Adam adam(0.1f);
    Tensor param({2});
    param.data()[0] = 1.0f;
    param.data()[1] = -2.0f;
    Tensor grad({2});
    grad.data()[0] = 0.5f;
    grad.data()[1] = -4.0f;

    adam.step(param, grad);

    EXPECT_NEAR(param.data()[0], 1.0f - 0.1f, 1e-6f);
    EXPECT_NEAR(param.data()[1], -2.0f + 0.1f, 1e-6f);
}

// A constant gradient keeps m_hat = g and v_hat = g² at every step, so three updates move the
// parameter by 3 * lr.
TEST(AdamTest, ConstantGradientMovesByLearningRateEachStep) {
    Adam adam(0.01f);
    Tensor param({1});
    Tensor grad({1});
    grad.data()[0] = 3.0f;
    for (int t = 0; t < 3; ++t) {
        adam.step(param, grad);
    }
    EXPECT_NEAR(param.data()[0], -0.03f, 1e-5f);
}

// Each parameter has its own step counter: updating one parameter must not change the bias
// correction used for another. A shared counter would make the second parameter's first
// update about 0.74 * lr instead of lr.
TEST(AdamTest, StepCountIsTrackedPerParameter) {
    Adam adam(0.1f);
    Tensor first({1});
    Tensor second({1});
    Tensor grad({1});
    grad.data()[0] = 1.0f;

    adam.step(first, grad);
    adam.step(second, grad);

    EXPECT_NEAR(first.data()[0], -0.1f, 1e-6f);
    EXPECT_NEAR(second.data()[0], -0.1f, 1e-6f);
}

TEST(AdamTest, ShapeMismatchThrows) {
    Adam adam;
    Tensor param({2});
    Tensor grad({3});
    EXPECT_THROW(adam.step(param, grad), std::invalid_argument);
}
