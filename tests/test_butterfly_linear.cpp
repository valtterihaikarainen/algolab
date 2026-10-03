#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vadugrad/butterfly_linear.hpp>

namespace {

Tensor dense_butterfly_matrix(const ButterflyLinear& layer) {
    const int n = layer.dim();
    Tensor dense({n, n});
    for (int j = 0; j < n; ++j) {
        Tensor e({1, n});
        e({0, j}) = 1.0f;
        ButterflyLinear tmp = layer.clone_weights();
        const Tensor col = tmp.forward(e);
        for (int i = 0; i < n; ++i) {
            dense({i, j}) = col({0, i});
        }
    }
    return dense;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    float m = 0.0f;
    for (int i = 0; i < a.numel(); ++i) {
        m = std::max(m, std::fabs(a.data()[i] - b.data()[i]));
    }
    return m;
}

}  // namespace

TEST(ButterflyLinearTest, InvalidDimThrows) {
    EXPECT_THROW(ButterflyLinear(3), std::invalid_argument);
    EXPECT_THROW(ButterflyLinear(0), std::invalid_argument);
}

TEST(ButterflyLinearTest, ForwardMatchesDenseConstruction) {
    ButterflyLinear bf(8);
    int t = 0;
    for (int s = 0; s < bf.num_stages(); ++s) {
        Tensor& w = bf.stage_weight(s);
        for (int k = 0; k < w.shape()[0]; ++k) {
            w({k, 0, 0}) = 0.1f * static_cast<float>(t++);
            w({k, 0, 1}) = 0.1f * static_cast<float>(t++);
            w({k, 1, 0}) = 0.1f * static_cast<float>(t++);
            w({k, 1, 1}) = 0.1f * static_cast<float>(t++);
        }
    }

    const Tensor dense = dense_butterfly_matrix(bf);

    Tensor x({2, 8});
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < 8; ++i) {
            x({b, i}) = 0.05f * static_cast<float>(b * 17 + i * 3 + 1);
        }
    }

    const Tensor y_bf = bf.forward(x);
    const Tensor dense_t = transpose2d(dense);
    const Tensor y_dn = matmul(x, dense_t);
    EXPECT_LT(max_abs_diff(y_bf, y_dn), 1e-4f);
}

TEST(ButterflyLinearTest, BackwardWeightsMatchFiniteDifferences) {
    ButterflyLinear bf(4);
    int t = 1;
    for (int s = 0; s < bf.num_stages(); ++s) {
        Tensor& w = bf.stage_weight(s);
        for (int k = 0; k < w.shape()[0]; ++k) {
            w({k, 0, 0}) = 0.07f * static_cast<float>(t++);
            w({k, 0, 1}) = -0.04f * static_cast<float>(t++);
            w({k, 1, 0}) = 0.11f * static_cast<float>(t++);
            w({k, 1, 1}) = 0.03f * static_cast<float>(t++);
        }
    }

    Tensor x({1, 4});
    x({0, 0}) = 0.2f;
    x({0, 1}) = -0.5f;
    x({0, 2}) = 0.9f;
    x({0, 3}) = 0.1f;

    Tensor gy({1, 4});
    gy({0, 0}) = 1.0f;
    gy({0, 1}) = -0.25f;
    gy({0, 2}) = 0.5f;
    gy({0, 3}) = 0.0f;

    (void)bf.forward(x);
    const auto g0 = bf.backward(gy);

    const float eps = 5e-3f;
    for (int s = 0; s < bf.num_stages(); ++s) {
        Tensor& w = bf.stage_weight(s);
        for (int k = 0; k < w.shape()[0]; ++k) {
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    const float saved = w({k, a, b});

                    w({k, a, b}) = saved + eps;
                    const Tensor yp = bf.forward(x);
                    float Lp = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        Lp += gy({0, i}) * yp({0, i});
                    }

                    w({k, a, b}) = saved - eps;
                    const Tensor ym = bf.forward(x);
                    float Lm = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        Lm += gy({0, i}) * ym({0, i});
                    }

                    w({k, a, b}) = saved;

                    const float fd = (Lp - Lm) / (2.0f * eps);
                    const float an = g0.grad_stages[static_cast<std::size_t>(s)]({k, a, b});
                    EXPECT_NEAR(fd, an, 2e-2f);
                }
            }
        }
    }
}
