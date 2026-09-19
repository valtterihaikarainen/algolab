/**
 * @file main.cpp
 * @brief Small CLI demo: build a @ref DenseLinear layer and print a forward/backward sanity check.
 */

#include <vadugrad/dense_linear.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            std::cout << "vadugrad — week 3 demo (dense linear forward/backward)\n";
            std::cout << "Usage: vadugrad [--help]\n";
            std::cout << "Runs a fixed 2→3 layer on one sample and prints y and dL/dx.\n";
            return 0;
        }
    }

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
