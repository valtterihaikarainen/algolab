#include <vadugrad/multihead_attention.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void validate_rank3_model(const Tensor& x, int d_model, const char* name) {
    if (x.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be rank-3 [batch, time, d_model]");
    }
    const auto s = x.shape();
    if (s[2] != d_model) {
        throw std::invalid_argument(std::string(name) + " last dimension must match d_model");
    }
}

Tensor flatten_bt(const Tensor& x) {
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
    const auto s = x.shape();
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
    const auto s = x.shape();  // [B, H, T, Dh]
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

}  // namespace

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, bool causal)
    : q_proj_(d_model, d_model),
      k_proj_(d_model, d_model),
      v_proj_(d_model, d_model),
      o_proj_(d_model, d_model),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(num_heads > 0 ? d_model / num_heads : 0),
      causal_(causal) {
    if (d_model <= 0 || num_heads <= 0) {
        throw std::invalid_argument("MultiHeadAttention: d_model and num_heads must be positive");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("MultiHeadAttention: d_model must be divisible by num_heads");
    }
}

int MultiHeadAttention::d_model() const { return d_model_; }
int MultiHeadAttention::num_heads() const { return num_heads_; }
int MultiHeadAttention::head_dim() const { return head_dim_; }
bool MultiHeadAttention::is_causal() const { return causal_; }

DenseLinear& MultiHeadAttention::q_proj() { return q_proj_; }
DenseLinear& MultiHeadAttention::k_proj() { return k_proj_; }
DenseLinear& MultiHeadAttention::v_proj() { return v_proj_; }
DenseLinear& MultiHeadAttention::o_proj() { return o_proj_; }

const DenseLinear& MultiHeadAttention::q_proj() const { return q_proj_; }
const DenseLinear& MultiHeadAttention::k_proj() const { return k_proj_; }
const DenseLinear& MultiHeadAttention::v_proj() const { return v_proj_; }
const DenseLinear& MultiHeadAttention::o_proj() const { return o_proj_; }

Tensor MultiHeadAttention::forward(const Tensor& x) {
    return forward(x, x);
}

Tensor MultiHeadAttention::forward(const Tensor& q_input, const Tensor& kv_input) {
    validate_rank3_model(q_input, d_model_, "q_input");
    validate_rank3_model(kv_input, d_model_, "kv_input");
    const auto sq = q_input.shape();
    const auto skv = kv_input.shape();
    if (sq[0] != skv[0]) {
        throw std::invalid_argument("MultiHeadAttention::forward: q_input and kv_input batch must match");
    }
    const int batch = sq[0];
    const int tq = sq[1];
    const int tk = skv[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    const Tensor q_lin = unflatten_bt(q_proj_.forward(flatten_bt(q_input)), batch, tq, d_model_);
    const Tensor k_lin = unflatten_bt(k_proj_.forward(flatten_bt(kv_input)), batch, tk, d_model_);
    const Tensor v_lin = unflatten_bt(v_proj_.forward(flatten_bt(kv_input)), batch, tk, d_model_);

    const Tensor qh = split_heads(q_lin, num_heads_, head_dim_);
    const Tensor kh = split_heads(k_lin, num_heads_, head_dim_);
    const Tensor vh = split_heads(v_lin, num_heads_, head_dim_);

    Tensor scores({batch, num_heads_, tq, tk});
    Tensor probs({batch, num_heads_, tq, tk});
    Tensor ctx({batch, num_heads_, tq, head_dim_});

    const bool apply_causal = causal_ && (tq == tk);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < tq; ++i) {
                float max_logit = -1e30f;
                for (int j = 0; j < tk; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) {
                        s += qh({b, h, i, d}) * kh({b, h, j, d});
                    }
                    s *= scale;
                    if (apply_causal && j > i) {
                        s = -1e30f;
                    }
                    scores({b, h, i, j}) = s;
                    if (s > max_logit) {
                        max_logit = s;
                    }
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < tk; ++j) {
                    const float e = std::exp(scores({b, h, i, j}) - max_logit);
                    probs({b, h, i, j}) = e;
                    sum_exp += e;
                }
                for (int j = 0; j < tk; ++j) {
                    probs({b, h, i, j}) /= sum_exp;
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < tq; ++i) {
                for (int d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < tk; ++j) {
                        acc += probs({b, h, i, j}) * vh({b, h, j, d});
                    }
                    ctx({b, h, i, d}) = acc;
                }
            }
        }
    }

    const Tensor ctx_concat = merge_heads(ctx, d_model_);
    const Tensor out = unflatten_bt(o_proj_.forward(flatten_bt(ctx_concat)), batch, tq, d_model_);

    cache_q_input_ = q_input;
    cache_kv_input_ = kv_input;
    cache_q_proj_ = q_lin;
    cache_k_proj_ = k_lin;
    cache_v_proj_ = v_lin;
    cache_attn_probs_ = probs;
    cache_context_concat_ = ctx_concat;
    return out;
}

MultiHeadAttention::BackwardGradients MultiHeadAttention::backward(const Tensor& grad_output) const {
    if (!cache_q_input_ || !cache_kv_input_ || !cache_q_proj_ || !cache_k_proj_ || !cache_v_proj_ ||
        !cache_attn_probs_ || !cache_context_concat_) {
        throw std::invalid_argument("MultiHeadAttention::backward called before forward");
    }
    validate_rank3_model(grad_output, d_model_, "grad_output");

    const auto sq = cache_q_input_->shape();
    const auto skv = cache_kv_input_->shape();
    const int batch = sq[0];
    const int tq = sq[1];
    const int tk = skv[1];
    if (grad_output.shape()[0] != batch || grad_output.shape()[1] != tq) {
        throw std::invalid_argument("MultiHeadAttention::backward: grad_output shape mismatch");
    }
    const bool apply_causal = causal_ && (tq == tk);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    const auto go2d = flatten_bt(grad_output);
    const auto ctx2d = flatten_bt(*cache_context_concat_);
    const auto go = o_proj_.backward(ctx2d, go2d);
    const Tensor grad_ctx_concat = unflatten_bt(go.grad_input, batch, tq, d_model_);

    const Tensor qh = split_heads(*cache_q_proj_, num_heads_, head_dim_);
    const Tensor kh = split_heads(*cache_k_proj_, num_heads_, head_dim_);
    const Tensor vh = split_heads(*cache_v_proj_, num_heads_, head_dim_);
    const Tensor probs = *cache_attn_probs_;
    const Tensor grad_ctx = split_heads(grad_ctx_concat, num_heads_, head_dim_);

    Tensor grad_probs({batch, num_heads_, tq, tk});
    Tensor grad_scores({batch, num_heads_, tq, tk});
    Tensor grad_qh({batch, num_heads_, tq, head_dim_});
    Tensor grad_kh({batch, num_heads_, tk, head_dim_});
    Tensor grad_vh({batch, num_heads_, tk, head_dim_});

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < tq; ++i) {
                for (int j = 0; j < tk; ++j) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) {
                        acc += grad_ctx({b, h, i, d}) * vh({b, h, j, d});
                        grad_vh({b, h, j, d}) += probs({b, h, i, j}) * grad_ctx({b, h, i, d});
                    }
                    grad_probs({b, h, i, j}) = acc;
                }

                float dot = 0.0f;
                for (int j = 0; j < tk; ++j) {
                    dot += grad_probs({b, h, i, j}) * probs({b, h, i, j});
                }
                for (int j = 0; j < tk; ++j) {
                    float ds = probs({b, h, i, j}) * (grad_probs({b, h, i, j}) - dot);
                    if (apply_causal && j > i) {
                        ds = 0.0f;
                    }
                    grad_scores({b, h, i, j}) = ds;
                }
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < tq; ++i) {
                for (int d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < tk; ++j) {
                        acc += grad_scores({b, h, i, j}) * kh({b, h, j, d});
                    }
                    grad_qh({b, h, i, d}) = acc * scale;
                }
            }
            for (int j = 0; j < tk; ++j) {
                for (int d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
                    for (int i = 0; i < tq; ++i) {
                        acc += grad_scores({b, h, i, j}) * qh({b, h, i, d});
                    }
                    grad_kh({b, h, j, d}) += acc * scale;
                }
            }
        }
    }

    const Tensor grad_q_proj = merge_heads(grad_qh, d_model_);
    const Tensor grad_k_proj = merge_heads(grad_kh, d_model_);
    const Tensor grad_v_proj = merge_heads(grad_vh, d_model_);

    const auto gq = q_proj_.backward(flatten_bt(*cache_q_input_), flatten_bt(grad_q_proj));
    const auto gk = k_proj_.backward(flatten_bt(*cache_kv_input_), flatten_bt(grad_k_proj));
    const auto gv = v_proj_.backward(flatten_bt(*cache_kv_input_), flatten_bt(grad_v_proj));

    Tensor grad_kv2d = gk.grad_input + gv.grad_input;

    return BackwardGradients{
        unflatten_bt(gq.grad_input, batch, tq, d_model_),
        unflatten_bt(grad_kv2d, batch, tk, d_model_),
        gq.grad_weight,
        gq.grad_bias,
        gk.grad_weight,
        gk.grad_bias,
        gv.grad_weight,
        gv.grad_bias,
        go.grad_weight,
        go.grad_bias};
}

const Tensor& MultiHeadAttention::attention_probs() const {
    if (!cache_attn_probs_) {
        throw std::invalid_argument("MultiHeadAttention::attention_probs called before forward");
    }
    return *cache_attn_probs_;
}
