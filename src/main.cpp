/**
 * @file main.cpp
 * @brief CLI demos for reviewers: dense linear layer and butterfly linear layer.
 */

#include <vadugrad/butterfly_linear.hpp>
#include <vadugrad/dense_linear.hpp>
#include <vadugrad/tensor.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    std::cout << "vadugrad — small manual-backprop demos\n";
    std::cout << "Usage:\n";
    std::cout << "  vadugrad              # dense linear demo (default)\n";
    std::cout << "  vadugrad butterfly    # butterfly linear demo (n=8)\n";
    std::cout << "  vadugrad --help\n";
}

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

int run_dense_demo() {
    // One sample, two inputs -> three outputs (hand-checkable with unit tests).
    DenseLinear layer(2, 3);
    Tensor& W = layer.weight();
    Tensor& b = layer.bias();

    W({0, 0}) = 1.0f;
    W({0, 1}) = 2.0f;
    W({0, 2}) = 3.0f;
    W({1, 0}) = 4.0f;
    W({1, 1}) = 5.0f;
    W({1, 2}) = 6.0f;

    b({0}) = 0.1f;
    b({1}) = 0.2f;
    b({2}) = 0.3f;

    Tensor x({1, 2});
    x({0, 0}) = 1.0f;
    x({0, 1}) = 2.0f;

    const Tensor y = layer.forward(x);
    std::cout << "vadugrad demo (batch=1, in=2, out=3)\n";
    std::cout << "y = [ " << y({0, 0}) << ", " << y({0, 1}) << ", " << y({0, 2}) << " ]\n";

    Tensor grad_y({1, 3});
    grad_y({0, 0}) = 1.0f;
    grad_y({0, 1}) = 0.0f;
    grad_y({0, 2}) = 0.0f;

    const auto g = layer.backward(x, grad_y);
    std::cout << "dL/dx = [ " << g.grad_input({0, 0}) << ", " << g.grad_input({0, 1}) << " ]\n";
    return 0;
}

int run_butterfly_demo() {
    constexpr int n = 8;
    ButterflyLinear bf(n);

    int t = 0;
    for (int s = 0; s < bf.num_stages(); ++s) {
        Tensor& w = bf.stage_weight(s);
        for (int k = 0; k < w.shape()[0]; ++k) {
            w({k, 0, 0}) = 0.05f * static_cast<float>(t++ + 1);
            w({k, 0, 1}) = -0.03f * static_cast<float>(t++ + 2);
            w({k, 1, 0}) = 0.07f * static_cast<float>(t++ + 3);
            w({k, 1, 1}) = 0.04f * static_cast<float>(t++ + 4);
        }
    }

    const Tensor dense = dense_butterfly_matrix(bf);

    Tensor x({1, n});
    for (int i = 0; i < n; ++i) {
        x({0, i}) = 0.1f * static_cast<float>(i + 1);
    }

    const Tensor y_bf = bf.forward(x);
    const Tensor dense_t = transpose2d(dense);
    const Tensor y_dn = matmul(x, dense_t);
    std::cout << "butterfly demo (n=" << n << ")\n";
    std::cout << "max|y_bf - y_dense| = " << max_abs_diff(y_bf, y_dn) << "\n";

    Tensor gy({1, n});
    gy.fill(1.0f);
    const auto g = bf.backward(gy);
    std::cout << "dL/dx[0] = " << g.grad_input({0, 0}) << "\n";
    std::cout << "example dL/dW stage0[k=0] = [ [" << g.grad_stages[0]({0, 0, 0}) << ", "
              << g.grad_stages[0]({0, 0, 1}) << "], [" << g.grad_stages[0]({0, 1, 0}) << ", "
              << g.grad_stages[0]({0, 1, 1}) << "] ]\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
        if (arg == "butterfly") {
            return run_butterfly_demo();
        }
    }

    return run_dense_demo();
}
