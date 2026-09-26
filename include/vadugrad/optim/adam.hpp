#pragma once

#include <unordered_map>
#include <vadugrad/tensor.hpp>

class Adam {
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    int step_;
    std::unordered_map<const float*, Tensor> m_;
    std::unordered_map<const float*, Tensor> v_;

public:
    Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step(Tensor& param, const Tensor& grad);
};
