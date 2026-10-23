#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vadugrad/butterfly_linear.hpp>

#include "grad_check.hpp"

namespace {

/** Fill every block of every stage and the bias with N(0, 1/2) samples. */
void randomize(ButterflyLinear& layer, unsigned int seed) {
    for (int s = 0; s < layer.num_stages(); ++s) {
        initialize_tensor_normal(layer.stage_weight(s), 0.7071f, seed);
    }
    initialize_tensor_normal(layer.bias(), 0.5f, seed);
}

/**
 * The n × n matrix M that the stages implement (bias excluded): column j is the layer's
 * output for the j-th unit vector.
 */
Tensor explicit_matrix(const ButterflyLinear& layer) {
    ButterflyLinear without_bias = layer;
    without_bias.bias().fill(0.0f);
    const int n = layer.dim();
    Tensor matrix({n, n});
    for (int j = 0; j < n; ++j) {
        Tensor unit({1, n});
        unit(0, j) = 1.0f;
        const Tensor column = without_bias.forward(unit);
        for (int i = 0; i < n; ++i) {
            matrix(i, j) = column(0, i);
        }
    }
    return matrix;
}

/** y = x M^T + b computed with an ordinary dense matrix product. */
Tensor dense_reference(const Tensor& x, const Tensor& matrix, const Tensor& bias) {
    Tensor y = matmul(x, transpose2d(matrix));
    for (int b = 0; b < y.shape()[0]; ++b) {
        for (int i = 0; i < y.shape()[1]; ++i) {
            y(b, i) += bias.data()[i];
        }
    }
    return y;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    float m = 0.0f;
    for (int i = 0; i < a.numel(); ++i) {
        m = std::max(m, std::fabs(a.data()[i] - b.data()[i]));
    }
    return m;
}

int popcount(int x) {
    int count = 0;
    for (; x != 0; x &= x - 1) {
        ++count;
    }
    return count;
}

/** Check input, stage and bias gradients of a random layer of width @p n against finite differences. */
void check_all_gradients(int n, unsigned int seed) {
    ButterflyLinear layer(n);
    randomize(layer, seed);
    Tensor x = grad_check::random_tensor({3, n}, seed + 1);
    const Tensor upstream = grad_check::random_tensor({3, n}, seed + 2);

    (void)layer.forward(x);
    const auto grads = layer.backward(upstream);
    auto loss = [&] { return grad_check::weighted_sum(layer.forward(x), upstream); };

    const std::string tag = "n=" + std::to_string(n);
    grad_check::expect_close(grads.grad_input, grad_check::numerical_gradient(x, loss), 1e-2f,
                             tag + " input");
    for (int s = 0; s < layer.num_stages(); ++s) {
        grad_check::expect_close(grads.grad_stages[static_cast<std::size_t>(s)],
                                 grad_check::numerical_gradient(layer.stage_weight(s), loss), 1e-2f,
                                 tag + " stage " + std::to_string(s));
    }
    grad_check::expect_close(grads.grad_bias, grad_check::numerical_gradient(layer.bias(), loss),
                             1e-2f, tag + " bias");
}

}  // namespace

TEST(ButterflyLinearTest, InvalidWidthThrows) {
    EXPECT_THROW(ButterflyLinear(0), std::invalid_argument);
    EXPECT_THROW(ButterflyLinear(1), std::invalid_argument);
    EXPECT_THROW(ButterflyLinear(3), std::invalid_argument);
    EXPECT_THROW(ButterflyLinear(784), std::invalid_argument);
    EXPECT_THROW(ButterflyLinear(-4), std::invalid_argument);
}

// The parameter counts reported in the experiment: log2(n) stages of n/2 blocks of 4 weights,
// plus n biases.
TEST(ButterflyLinearTest, ParameterCountMatchesFormula) {
    EXPECT_EQ(ButterflyLinear(2).num_parameters(), 1 * 1 * 4 + 2);
    EXPECT_EQ(ButterflyLinear(8).num_parameters(), 3 * 4 * 4 + 8);
    EXPECT_EQ(ButterflyLinear(512).num_parameters(), 9 * 256 * 4 + 512);
}

// Smallest case worked out by hand: n = 2 has one stage with a single block, so the layer is
// exactly y = W x + b.
TEST(ButterflyLinearTest, SingleBlockMatchesHandComputation) {
    ButterflyLinear layer(2);
    Tensor& w = layer.stage_weight(0);
    w(0, 0, 0) = 1.0f;
    w(0, 0, 1) = 2.0f;
    w(0, 1, 0) = 3.0f;
    w(0, 1, 1) = 4.0f;
    layer.bias().data()[0] = 0.5f;
    layer.bias().data()[1] = -0.5f;

    Tensor x({1, 2});
    x(0, 0) = 1.0f;
    x(0, 1) = -1.0f;
    const Tensor y = layer.forward(x);
    EXPECT_FLOAT_EQ(y(0, 0), 1.0f - 2.0f + 0.5f);
    EXPECT_FLOAT_EQ(y(0, 1), 3.0f - 4.0f - 0.5f);
}

// Checks the stage wiring. With every block set to [[1, 1], [1, -1]], the butterfly is the
// fast Walsh–Hadamard transform, whose matrix is H[i][j] = (-1)^popcount(i AND j). Any
// mistake in which positions are paired, or in which block a pair uses, breaks this.
TEST(ButterflyLinearTest, HadamardBlocksGiveWalshHadamardTransform) {
    for (const int n : {8, 512}) {
        ButterflyLinear layer(n);
        for (int s = 0; s < layer.num_stages(); ++s) {
            Tensor& w = layer.stage_weight(s);
            for (int k = 0; k < n / 2; ++k) {
                w(k, 0, 0) = 1.0f;
                w(k, 0, 1) = 1.0f;
                w(k, 1, 0) = 1.0f;
                w(k, 1, 1) = -1.0f;
            }
        }
        const Tensor matrix = explicit_matrix(layer);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float expected = popcount(i & j) % 2 == 0 ? 1.0f : -1.0f;
                ASSERT_FLOAT_EQ(matrix(i, j), expected) << "n=" << n << " i=" << i << " j=" << j;
            }
        }
    }
}

// After log2(n) stages every output depends on every input, like a dense layer. With random
// blocks, every entry of the implemented matrix is therefore non-zero.
TEST(ButterflyLinearTest, EveryOutputDependsOnEveryInput) {
    ButterflyLinear layer(64);
    randomize(layer, 11u);
    const Tensor matrix = explicit_matrix(layer);
    for (int i = 0; i < matrix.numel(); ++i) {
        EXPECT_NE(matrix.data()[i], 0.0f) << "entry " << i;
    }
}

// The batched forward pass must agree with multiplying by the explicit matrix, at the width
// used in the experiment (n = 512, 9 stages) and with several rows to catch batch-indexing bugs.
TEST(ButterflyLinearTest, ForwardMatchesExplicitMatrixAtExperimentWidth) {
    constexpr int n = 512;
    ButterflyLinear layer(n);
    randomize(layer, 5u);
    const Tensor x = grad_check::random_tensor({4, n}, 6u);
    const Tensor expected = dense_reference(x, explicit_matrix(layer), layer.bias());
    EXPECT_LT(max_abs_diff(layer.forward(x), expected), 1e-3f);
}

TEST(ButterflyLinearTest, GradientsMatchFiniteDifferencesSmall) { check_all_gradients(8, 21u); }

TEST(ButterflyLinearTest, GradientsMatchFiniteDifferencesMedium) { check_all_gradients(64, 31u); }

TEST(ButterflyLinearTest, BackwardBeforeForwardThrows) {
    ButterflyLinear layer(4);
    EXPECT_THROW((void)layer.backward(Tensor({1, 4})), std::invalid_argument);
}

TEST(ButterflyLinearTest, WrongShapesThrow) {
    ButterflyLinear layer(4);
    EXPECT_THROW((void)layer.forward(Tensor({1, 8})), std::invalid_argument);
    EXPECT_THROW((void)layer.forward(Tensor({4})), std::invalid_argument);
    (void)layer.forward(Tensor({2, 4}));
    EXPECT_THROW((void)layer.backward(Tensor({3, 4})), std::invalid_argument);
}
