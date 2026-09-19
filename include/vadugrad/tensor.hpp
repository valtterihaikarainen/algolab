#pragma once

/**
 * @file tensor.hpp
 * @brief Row-major @c float tensor used as storage for the MNIST MLP (see specification).
 *
 * Memory layout is contiguous C-style (last index varies fastest). Strides follow NumPy-style
 * row-major semantics: @c offset = sum_i idx[i] * strides[i].
 */

#include <initializer_list>
#include <stdexcept>
#include <vector>

/**
 * @brief Compute row-major strides for a shape.
 *
 * For shape @f$[d_0,\ldots,d_{n-1}]@f$, returns @f$[s_0,\ldots,s_{n-1}]@f$ with
 * @f$s_{n-1}=1@f$ and @f$s_i = s_{i+1}\cdot d_{i+1}@f$.
 *
 * @param shape Tensor dimensions (each dimension must be positive for a valid tensor).
 * @return Strides of the same length as @p shape.
 */
std::vector<int> compute_strides(const std::vector<int>& shape);

/**
 * @brief Multi-dimensional array of @c float with owned storage.
 *
 * Owns a heap buffer for data, shape, and strides. Copy operations deep-copy; move operations
 * transfer ownership and leave the source in an empty state (@c ndim()==0, null data).
 */
class Tensor {
    float* data_;    ///< Contiguous row-major data, length @c numel_.
    int* shape_;     ///< Dimension sizes.
    int* strides_;   ///< Row-major strides (same length as @c shape_).
    int ndim_;       ///< Number of dimensions.
    int numel_;      ///< Product of shape entries; size of @c data_.

public:
    /**
     * @brief Construct a tensor with the given shape.
     *
     * Allocates contiguous row-major storage and zero-initializes all elements.
     *
     * @param shape Dimensions; each entry must be positive for the intended use in this project.
     */
    explicit Tensor(const std::vector<int>& shape);

    /** @brief Release owned buffers. */
    ~Tensor();

    /**
     * @brief Deep copy: new allocation with the same shape and element values.
     * @param other Source tensor.
     */
    Tensor(const Tensor& other);

    /**
     * @brief Deep copy assignment: replaces contents with a copy of @p other.
     * @param other Source tensor.
     * @return Reference to @c *this.
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Move constructor: takes ownership; @p other is left empty but valid.
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief Move assignment: releases current buffers, then takes ownership from @p other.
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /** @return Number of dimensions. */
    int ndim() const;

    /** @return Total number of elements (product of shape). */
    int numel() const;

    /** @return A copy of the shape vector. */
    std::vector<int> shape() const;

    /** @return A copy of the row-major strides. */
    std::vector<int> strides() const;

    /** @return Mutable pointer to contiguous storage (length @c numel()). */
    float* data();

    /** @return Read-only pointer to contiguous storage. */
    const float* data() const;

    /**
     * @brief Map a multi-index to a flat offset in row-major order.
     *
     * @param idx Index along each dimension; length must equal @c ndim().
     * @return Offset in @c [0, numel()) suitable for @c data()[offset].
     * @throws std::invalid_argument if @c idx.size() != ndim().
     * @throws std::out_of_range if any index is outside its dimension.
     */
    int offset(const std::vector<int>& idx) const;

    /** @brief Element access by multi-index (mutable). */
    float& at(const std::vector<int>& idx);

    /** @brief Element access by multi-index (const). */
    const float& at(const std::vector<int>& idx) const;

    /** @brief Element access using an initializer list as multi-index, e.g. @c t({0,1,2}). */
    float& operator()(std::initializer_list<int> idx);

    /** @brief Const element access using an initializer list as multi-index. */
    const float& operator()(std::initializer_list<int> idx) const;

    /**
     * @brief Set every element to @p v.
     * @param v Fill value.
     */
    void fill(float v);

    /**
     * @brief Change shape without changing element count or storage order.
     *
     * @param new_shape New dimensions; product must equal @c numel().
     * @throws std::invalid_argument if the product differs from @c numel() or any dimension is
     *         non-positive.
     */
    void reshape(const std::vector<int>& new_shape);
};

/**
 * @brief Matrix multiplication for rank-2 tensors.
 *
 * Computes @f$C = A \times B@f$ for row-major 2D tensors.
 *
 * @param a Left operand with shape [m, k].
 * @param b Right operand with shape [k, n].
 * @return Tensor with shape [m, n].
 *
 * @throws std::invalid_argument if either tensor is not rank-2 or inner dimensions mismatch.
 */
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Elementwise addition of two tensors with identical shape.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Elementwise subtraction of two tensors with identical shape.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor sub(const Tensor& a, const Tensor& b);

/**
 * @brief Elementwise multiplication of two tensors with identical shape.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief Elementwise division of two tensors with identical shape.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor div(const Tensor& a, const Tensor& b);

/**
 * @brief Add scalar @p s to each element of @p a.
 */
[[nodiscard]] Tensor add(const Tensor& a, float s);

/**
 * @brief Subtract scalar @p s from each element of @p a.
 */
[[nodiscard]] Tensor sub(const Tensor& a, float s);

/**
 * @brief Multiply each element of @p a by scalar @p s.
 */
[[nodiscard]] Tensor mul(const Tensor& a, float s);

/**
 * @brief Divide each element of @p a by scalar @p s.
 */
[[nodiscard]] Tensor div(const Tensor& a, float s);

/**
 * @brief Convenience operator for elementwise tensor addition.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor operator+(const Tensor& a, const Tensor& b);

/**
 * @brief Convenience operator for elementwise tensor subtraction.
 * @throws std::invalid_argument if shapes differ.
 */
[[nodiscard]] Tensor operator-(const Tensor& a, const Tensor& b);

/**
 * @brief Convenience operator for tensor-scalar multiplication.
 */
[[nodiscard]] Tensor operator*(const Tensor& a, float s);

/**
 * @brief Convenience operator for tensor-scalar multiplication.
 */
[[nodiscard]] Tensor operator*(float s, const Tensor&a);

 /**
 * @brief Transpose a rank-2 tensor.
 *
 * If input has shape [m, n], returns shape [n, m].
 *
 * @param x Input tensor.
 * @return Transposed tensor with copied data (no view semantics).
 *
 * @throws std::invalid_argument if @p x.ndim() != 2.
 */
[[nodiscard]] Tensor transpose2d(const Tensor& x);

/**
 * @brief Reduce tensor by summing along one axis.
 *
 * @param x Input tensor.
 * @param axis Axis to reduce in range [0, x.ndim()).
 * @param keepdim If true, reduced axis is kept with size 1; otherwise removed.
 * @return Reduced tensor.
 *
 * @throws std::invalid_argument if axis is out of range.
 */
[[nodiscard]] Tensor sum(const Tensor& x, int axis, bool keepdim = false);

