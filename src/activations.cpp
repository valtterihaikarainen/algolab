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
    constexpr float k = 0.7978845608f;   // sqrt(2/pi)
    constexpr float c = 0.044715f;
    for (int i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float u = k * (v + c * v * v * v);
        out.data()[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
    return out;
}

Tensor gelu_backward(const Tensor& x, const Tensor& grad_output) {
    if (x.shape() != grad_output.shape()) {
        throw std::invalid_argument("gelu_backward shape mismatch");
    }
    Tensor out(x.shape());
    constexpr float k = 0.7978845608f;   // sqrt(2/pi)
    constexpr float c = 0.044715f;
    for (int i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float u = k * (v + c * v * v * v);
        const float th = std::tanh(u);
        const float sech2 = 1.0f - th * th;
        const float du = k * (1.0f + 3.0f * c * v * v);
        const float dgelu = 0.5f * (1.0f + th) + 0.5f * v * sech2 * du;
        out.data()[i] = grad_output.data()[i] * dgelu;
    }
    return out;
}
