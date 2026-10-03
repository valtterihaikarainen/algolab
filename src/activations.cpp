#include <vadugrad/nn/activations.hpp>

#include <cmath>
#include <stdexcept>

Tensor relu(const Tensor& x) {
    Tensor out(x.shape());
    for (int i = 0; i < x.numel(); ++i) {
        out.data()[i] = x.data()[i] > 0.0f ? x.data()[i] : 0.0f;
    }
    return out;
}

Tensor relu_backward(const Tensor& x, const Tensor& grad_output) {
    if (x.shape() != grad_output.shape()) {
        throw std::invalid_argument("relu_backward shape mismatch");
    }
    Tensor out(x.shape());
    for (int i = 0; i < x.numel(); ++i) {
        out.data()[i] = x.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape());
    // tanh approximation from Hendrycks & Gimpel: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    constexpr float kGeluCubicCoeff = 0.044715f;
    for (int i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float u = kSqrtTwoOverPi * (v + kGeluCubicCoeff * v * v * v);
        out.data()[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
    return out;
}

Tensor gelu_backward(const Tensor& x, const Tensor& grad_output) {
    if (x.shape() != grad_output.shape()) {
        throw std::invalid_argument("gelu_backward shape mismatch");
    }
    Tensor out(x.shape());
    // Derivative of the same tanh GELU approximation used in gelu().
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    constexpr float kGeluCubicCoeff = 0.044715f;
    for (int i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float u = kSqrtTwoOverPi * (v + kGeluCubicCoeff * v * v * v);
        const float th = std::tanh(u);
        const float sech2 = 1.0f - th * th;
        const float du = kSqrtTwoOverPi * (1.0f + 3.0f * kGeluCubicCoeff * v * v);
        const float dgelu = 0.5f * (1.0f + th) + 0.5f * v * sech2 * du;
        out.data()[i] = grad_output.data()[i] * dgelu;
    }
    return out;
}
