#pragma once

#include <vadugrad/tensor.hpp>

[[nodiscard]] Tensor flatten_bt(const Tensor& x);
[[nodiscard]] Tensor unflatten_bt(const Tensor& x2d, int batch, int time, int d_model);
[[nodiscard]] Tensor split_heads(const Tensor& x, int num_heads, int head_dim);
[[nodiscard]] Tensor merge_heads(const Tensor& x, int d_model);
[[nodiscard]] Tensor add3d(const Tensor& a, const Tensor& b);
