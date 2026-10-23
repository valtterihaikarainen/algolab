/**
 * @file tensor.cpp
 * @brief Implementation of @ref Tensor and the matrix helpers. The public API is documented
 *        in tensor.hpp.
 */

#include <vadugrad/tensor.hpp>
#include <stdexcept>
#include <vector>

std::vector<int> compute_strides(const std::vector<int>& shape) {

    if (shape.empty()) {
        throw std::invalid_argument("shape cant be empty");
    }

    // A stride is how many floats to skip to move one step along a dimension. In row-major
    // order the last dimension is contiguous (stride 1), and each earlier stride is the next
    // one times the next dimension's size: shape [2, 3, 4] gives strides [12, 4, 1].
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

// The tensor owns three raw arrays, so copying must allocate new ones (a deep copy). Moving
// instead hands the arrays over and leaves the source empty, which avoids copying large
// batches when tensors are returned from functions.
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

Tensor transpose2d(const Tensor& x) {

    // Validating the rank
    if (x.ndim() != 2) {
        throw std::invalid_argument("The number of dimensions should equal 2");
    }

    std::vector<int> shape = x.shape();
    const int rows = shape[0];
    const int columns = shape[1];

    Tensor output({columns, rows});

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            output(j, i) = x(i, j);
        }
    }
    return output;
}

Tensor matmul(const Tensor& a, const Tensor& b) {

    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul expects rank-2 tensors");
    }

    const std::vector<int> shape_a = a.shape();
    const int m = shape_a[0];
    const int k = shape_a[1];
    
    const std::vector<int> shape_b = b.shape();
    const int k2 = shape_b[0];
    const int n = shape_b[1];

    if (k != k2) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }

    Tensor output({m, n});
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = output.data();

    // i-p-j loop order walks both b and output row by row, so the inner loop reads
    // contiguous memory and the compiler can vectorize it.
    for (int i = 0; i < m; ++i) {
        float* out_row = out_data + i * n;
        for (int p = 0; p < k; ++p) {
            const float a_ip = a_data[i * k + p];
            const float* b_row = b_data + p * n;
            for (int j = 0; j < n; ++j) {
                out_row[j] += a_ip * b_row[j];
            }
        }
    }
    return output;
}

Tensor sum(const Tensor& x, int axis, bool keepdim) {

    const int ndim = x.ndim();
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument("axis needs to be within [0, x.ndim]");
    }

    const auto shape = x.shape();

    // Build output shape
    std::vector<int> out_shape;
    out_shape.reserve(keepdim ? ndim : ndim - 1);
    for (int d = 0; d < ndim; ++d) {
        if (d == axis) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[d]);
        }
    }

    if (out_shape.empty()) {
        throw std::invalid_argument("sum producing rank-0 tensor is not supported");
    }

    Tensor out(out_shape);

    // View the row-major data as a 3D block [outer, reduce, inner]: outer is the product of
    // the dimensions before the axis, reduce the axis itself, inner the product of those after
    // it. Element (o, r, i) then sits at flat index (o * reduce + r) * inner + i, and summing
    // over r gives output element (o, i). For a [B, n] tensor and axis 0 this is the column
    // sum used for bias gradients (outer = 1, reduce = B, inner = n).
    int outer = 1;
    for (int d = 0; d < axis; ++d) outer *= shape[d];

    const int reduce = shape[axis];

    int inner = 1;
    for (int d = axis +1; d < static_cast<int>(shape.size()); ++d) inner *= shape[d];

    const float* xd = x.data();
    float* od = out.data();

    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < reduce; ++r) {
                int in_flat = (o * reduce + r) * inner + i;
                acc += xd[in_flat];
            }
            int out_flat = o * inner + i;
            od[out_flat] = acc;
        }
    }

    return out;
}
