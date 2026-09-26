# Testing

This document describes how `vadugrad` is tested, what has been tested, and how tests can be reproduced.

## 1. Framework and test types

- **Unit testing framework**: [GoogleTest](https://github.com/google/googletest)
- **Test categories in this repository**:
  - unit tests for tensor primitives and math ops
  - unit tests for neural network layers/modules
  - small integration tests for end-to-end training behavior

Not all possible test types are needed in this project. The main correctness risks here are:

1. shape/axis mistakes
2. backward gradient routing mistakes
3. numerical instability in softmax/cross-entropy path
4. integration mismatch between model backward output and optimizer update

Current test suite is designed specifically around those risks.

## 2. How to run tests

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cd build && ctest --output-on-failure
```

Alternative:

```bash
./build/tests
```

## 3. What is tested and with what inputs

### Tensor core (`tests/test_tensor.cpp`)

Tested:

- strides calculation (`compute_strides`)
- copy/move/assignment behavior
- fill, reshape invariants, index out-of-range handling
- elementwise ops, transpose, matmul, sum

Inputs:

- small hand-constructed tensors (`2x2`, `2x3`, `3x2`, simple 3D shapes)
- boundary/error cases (bad ranks, bad axis, mismatching shapes)

### Dense layer (`tests/test_dense_linear.cpp`)

Tested:

- forward against hand-computed values
- backward gradients (`dL/dx`, `dL/dW`, `dL/db`) against hand-computed values
- batched gradient aggregation behavior
- constructor/input validation errors

Inputs:

- small deterministic matrices/vectors with exact expected outputs

### Attention and transformer modules

- `tests/test_multihead_attention.cpp`
  - constructor validation
  - self/cross forward shape checks
  - backward shape checks
  - backward-before-forward error
  - causal mask behavior (future positions zeroed)
- `tests/test_layer_norm.cpp`
  - forward/backward shape checks
- `tests/test_nn_ops.cpp`
  - cross-entropy output and gradient shape checks
- `tests/test_decoder_only_transformer.cpp`
  - decoder model forward/backward shape checks across full stack

Inputs:

- small synthetic batches with fixed tensor dimensions
- deterministic token-id tensors for reproducible checks

### Integration test (`tests/test_tiny_overfit.cpp`)

Tested:

- end-to-end training loop signal: loss should decrease on a tiny fixed batch

Inputs:

- one tiny token sequence batch
- fixed target next-token labels
- fixed random seed for parameter initialization

This is an empirical test that complements unit tests by validating module interoperability.

## 4. Coverage collection (gcov/lcov)

Configure with coverage flags:

```bash
cmake -S . -B build-coverage -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-coverage
cd build-coverage && ctest --output-on-failure
```

Generate gcov output (example):

```bash
gcov -b -s ../src \
  CMakeFiles/vadugrad.dir/src/tensor.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/dense_linear.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/multihead_attention.cpp.gcno
```

Optional HTML report with lcov/genhtml (see `README.md`).

## 5. Testing limitations and future improvements

- Current suite emphasizes shape correctness and integration flow more than exact numerical gradient checking.
- Next useful extension: finite-difference gradient checks for selected modules (`LayerNorm`, `FeedForward`, `TransformerBlock`).
- Performance benchmarks are not yet part of automated tests.
