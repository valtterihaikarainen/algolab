/**
 * @file tensor.cpp
 * @brief Implementation of @ref Tensor and @ref compute_strides.
 */

#include <vadugrad/tensor.hpp>
#include <stdexcept>
#include <vector>

std::vector<int> compute_strides(const std::vector<int>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<int> strides(ndim);
    strides[ndim - 1] = 1;

    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

Tensor::Tensor(const std::vector<int>& shape) {
    ndim_ = static_cast<int>(shape.size());

    numel_ = 1;
    for (int s : shape) numel_ *= s;

    shape_ = new int[ndim_];
    for (int i = 0; i < ndim_; i++) shape_[i] = shape[i];

    std::vector<int> strides_vec = compute_strides(shape);
    strides_ = new int[ndim_];
    for (int i = 0; i < ndim_; i++) strides_[i] = strides_vec[i];

    data_ = new float[numel_]();
}

Tensor::~Tensor() {
    delete[] data_;
    delete[] shape_;
    delete[] strides_;
}

Tensor::Tensor(const Tensor& other) {
    ndim_ = other.ndim_;
    numel_ = other.numel_;

    data_ = new float[numel_];
    shape_ = new int[ndim_];
    strides_ = new int[ndim_];

    for (int i = 0; i < numel_; i++) data_[i] = other.data_[i];
    for (int i = 0; i < ndim_; i++) shape_[i] = other.shape_[i];
    for (int i = 0; i < ndim_; i++) strides_[i] = other.strides_[i];
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    delete[] data_;
    delete[] shape_;
    delete[] strides_;

    ndim_ = other.ndim_;
    numel_ = other.numel_;

    data_ = new float[numel_];
    shape_ = new int[ndim_];
    strides_ = new int[ndim_];

    for (int i = 0; i < numel_; i++) data_[i] = other.data_[i];
    for (int i = 0; i < ndim_; i++) shape_[i] = other.shape_[i];
    for (int i = 0; i < ndim_; i++) strides_[i] = other.strides_[i];

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept 
        : data_(other.data_),
          shape_(other.shape_),
          strides_(other.strides_),
          ndim_(other.ndim_),
          numel_(other.numel_) {
        other.data_ = nullptr;
        other.shape_ = nullptr;
        other.strides_ = nullptr;
        other.ndim_ = 0;
        other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    delete[] data_;
    delete[] shape_;
    delete[] strides_;

    data_ = other.data_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    ndim_ = other.ndim_;
    numel_ = other.numel_;

    other.data_ = nullptr;
    other.shape_ = nullptr;
    other.strides_ = nullptr;
    other.ndim_ = 0;
    other.numel_ = 0;

    return *this;
}

int Tensor::ndim() const {
    return ndim_;
}

int Tensor::numel() const {
    return numel_;
}

std::vector<int> Tensor::shape() const {
    if (shape_ == nullptr || ndim_ == 0) return {};
    return std::vector<int>(shape_, shape_ + ndim_);
}

std::vector<int> Tensor::strides() const {
    if (!strides_ || ndim_ == 0) return {};
    return std::vector<int>(strides_, strides_ + ndim_);
}

float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

int Tensor::offset(const std::vector<int>& idx) const {
    
    if (static_cast<int>(idx.size()) != ndim_) {
        throw std::invalid_argument("idx rank must match tensor ndim");
    }
    int off = 0;
    for (int i = 0; i < ndim_; i++) {
        if (idx[i] < 0 || idx[i] >= shape_[i]) {
            throw std::out_of_range("index out of bounds");
        }
        off += idx[i] * strides_[i];
    }
    return off; 
}

float& Tensor::at(const std::vector<int>& idx) {
    return data_[offset(idx)];
}

const float& Tensor::at(const std::vector<int>& idx) const {
    return data_[offset(idx)];
}

float& Tensor::operator()(std::initializer_list<int> idx) {
    return at(std::vector<int>(idx));
}

const float& Tensor::operator()(std::initializer_list<int> idx) const {
    return at(std::vector<int>(idx));
}

void Tensor::fill(float v) {
    for (int i = 0; i < numel_; ++i) {
        data_[i] = v;
    }
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    int new_ndim = static_cast<int>(new_shape.size());
    int new_numel = 1;
    for (int d : new_shape) {
        if (d <= 0 ) throw std::invalid_argument("Each dimension needs to be positive");
        new_numel *= d;
    }
    if (new_numel != numel_) {
        throw std::invalid_argument("Reshape must preserve number of elements");
    }

    std::vector<int> new_strides = compute_strides(new_shape);

    delete[] shape_;
    delete[] strides_;

    shape_ = new int[new_ndim];
    strides_ = new int[new_ndim];

    for (int i = 0; i < new_ndim; ++i) {
        shape_[i] = new_shape[i];
        strides_[i] = new_strides[i];
    }

    ndim_ = new_ndim;
}
















