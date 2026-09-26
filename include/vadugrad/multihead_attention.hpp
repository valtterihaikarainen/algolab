#pragma once

#include <vadugrad/dense_linear.hpp>
#include <vadugrad/tensor.hpp>
#include <optional>

/**
 * @file multihead_attention.hpp
 * @brief Multi-head attention (self and cross) with manual backprop.
 */

class MultiHeadAttention {
    DenseLinear q_proj_;  ///< Query projection d_model -> d_model.
    DenseLinear k_proj_;  ///< Key projection d_model -> d_model.
    DenseLinear v_proj_;  ///< Value projection d_model -> d_model.
    DenseLinear o_proj_;  ///< Output projection d_model -> d_model.
    int d_model_;
    int num_heads_;
    int head_dim_;
    bool causal_;

    std::optional<Tensor> cache_q_input_;
    std::optional<Tensor> cache_kv_input_;
    std::optional<Tensor> cache_q_proj_;
    std::optional<Tensor> cache_k_proj_;
    std::optional<Tensor> cache_v_proj_;
    std::optional<Tensor> cache_attn_probs_;
    std::optional<Tensor> cache_context_concat_;

public:
    /**
     * @param d_model Embedding/model width.
     * @param num_heads Number of attention heads (must divide @p d_model).
     * @param causal If true, applies upper-triangular causal masking when Tq == Tk.
     */
    MultiHeadAttention(int d_model, int num_heads, bool causal = false);

    int d_model() const;
    int num_heads() const;
    int head_dim() const;
    bool is_causal() const;

    /** @return Mutable Q projection layer. */
    DenseLinear& q_proj();
    /** @return Mutable K projection layer. */
    DenseLinear& k_proj();
    /** @return Mutable V projection layer. */
    DenseLinear& v_proj();
    /** @return Mutable output projection layer. */
    DenseLinear& o_proj();

    const DenseLinear& q_proj() const;
    const DenseLinear& k_proj() const;
    const DenseLinear& v_proj() const;
    const DenseLinear& o_proj() const;

    /**
     * @brief Self-attention convenience wrapper; equivalent to @c forward(x, x).
     * @param x Shape [batch, T, d_model].
     */
    [[nodiscard]] Tensor forward(const Tensor& x);

    /**
     * @brief General attention. Query from @p q_input, key/value from @p kv_input.
     *
     * @param q_input [batch, Tq, d_model]
     * @param kv_input [batch, Tk, d_model]
     * @return [batch, Tq, d_model]
     */
    [[nodiscard]] Tensor forward(const Tensor& q_input, const Tensor& kv_input);

    struct BackwardGradients {
        Tensor grad_q_input;   ///< [batch, Tq, d_model]
        Tensor grad_kv_input;  ///< [batch, Tk, d_model]
        Tensor grad_Wq;        ///< [d_model, d_model]
        Tensor grad_bq;        ///< [d_model]
        Tensor grad_Wk;        ///< [d_model, d_model]
        Tensor grad_bk;        ///< [d_model]
        Tensor grad_Wv;        ///< [d_model, d_model]
        Tensor grad_bv;        ///< [d_model]
        Tensor grad_Wo;        ///< [d_model, d_model]
        Tensor grad_bo;        ///< [d_model]
    };

    /**
     * @brief Backward pass for the latest @ref forward call.
     * @param grad_output Upstream gradient [batch, Tq, d_model].
     */
    [[nodiscard]] BackwardGradients backward(const Tensor& grad_output) const;

    /** @return Latest attention probabilities [batch, heads, Tq, Tk]. */
    const Tensor& attention_probs() const;
};