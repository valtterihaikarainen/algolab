#include <vadugrad/optim/adam.hpp>

#include <cmath>
#include <stdexcept>

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), step_(0) {}

void Adam::step(Tensor& param, const Tensor& grad) {
    if (param.shape() != grad.shape()) {
        throw std::invalid_argument("Adam::step param and grad shape mismatch");
    }
    const float* key = param.data();
    auto it_m = m_.find(key);
    auto it_v = v_.find(key);
    if (it_m == m_.end()) {
        auto m_ins = m_.emplace(key, Tensor(param.shape()));
        auto v_ins = v_.emplace(key, Tensor(param.shape()));
        m_ins.first->second.fill(0.0f);
        v_ins.first->second.fill(0.0f);
        it_m = m_ins.first;
        it_v = v_ins.first;
    }
    ++step_;
    Tensor& m = it_m->second;
    Tensor& v = it_v->second;
    const float b1_correction = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float b2_correction = 1.0f - std::pow(beta2_, static_cast<float>(step_));

    for (int i = 0; i < param.numel(); ++i) {
        const float g = grad.data()[i];
        m.data()[i] = beta1_ * m.data()[i] + (1.0f - beta1_) * g;
        v.data()[i] = beta2_ * v.data()[i] + (1.0f - beta2_) * g * g;
        const float m_hat = m.data()[i] / b1_correction;
        const float v_hat = v.data()[i] / b2_correction;
        param.data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
}
