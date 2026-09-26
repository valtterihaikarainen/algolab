#include <vadugrad/core/tensor_ops.hpp>

#include <stdexcept>

Tensor flatten_bt(const Tensor& x) {
    if (x.ndim() != 3) {
        throw std::invalid_argument("flatten_bt expects rank-3 [B,T,D]");
    }
    const auto s = x.shape();
    Tensor out({s[0] * s[1], s[2]});
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            const int row = b * s[1] + t;
            for (int d = 0; d < s[2]; ++d) {
                out({row, d}) = x({b, t, d});
            }
        }
    }
    return out;
}

Tensor unflatten_bt(const Tensor& x2d, int batch, int time, int d_model) {
    if (x2d.ndim() != 2 || x2d.shape()[0] != batch * time || x2d.shape()[1] != d_model) {
        throw std::invalid_argument("unflatten_bt shape mismatch");
    }
    Tensor out({batch, time, d_model});
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < time; ++t) {
            const int row = b * time + t;
            for (int d = 0; d < d_model; ++d) {
                out({b, t, d}) = x2d({row, d});
            }
        }
    }
    return out;
}

Tensor split_heads(const Tensor& x, int num_heads, int head_dim) {
    if (x.ndim() != 3) {
        throw std::invalid_argument("split_heads expects rank-3 [B,T,D]");
    }
    const auto s = x.shape();
    if (s[2] != num_heads * head_dim) {
        throw std::invalid_argument("split_heads last dim mismatch");
    }
    Tensor out({s[0], num_heads, s[1], head_dim});
    for (int b = 0; b < s[0]; ++b) {
        for (int t = 0; t < s[1]; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    out({b, h, t, d}) = x({b, t, h * head_dim + d});
                }
            }
        }
    }
    return out;
}

Tensor merge_heads(const Tensor& x, int d_model) {
    if (x.ndim() != 4) {
        throw std::invalid_argument("merge_heads expects rank-4 [B,H,T,Dh]");
    }
    const auto s = x.shape();
    if (s[1] * s[3] != d_model) {
        throw std::invalid_argument("merge_heads d_model mismatch");
    }
    Tensor out({s[0], s[2], d_model});
    for (int b = 0; b < s[0]; ++b) {
        for (int h = 0; h < s[1]; ++h) {
            for (int t = 0; t < s[2]; ++t) {
                for (int d = 0; d < s[3]; ++d) {
                    out({b, t, h * s[3] + d}) = x({b, h, t, d});
                }
            }
        }
    }
    return out;
}

Tensor add3d(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 3 || b.ndim() != 3 || a.shape() != b.shape()) {
        throw std::invalid_argument("add3d expects equal rank-3 tensors");
    }
    Tensor out(a.shape());
    const int n = a.numel();
    for (int i = 0; i < n; ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
    return out;
}
