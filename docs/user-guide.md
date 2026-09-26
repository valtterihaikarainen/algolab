# User Guide

## 1. Requirements

- CMake >= 3.20
- C++17 compiler
- Network access for first configure (GoogleTest is fetched by CMake)

## 2. Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## 3. Running executables

### `vadugrad` (basic CLI demo)

```bash
./build/vadugrad
./build/vadugrad --help
```

This runs a small dense-layer forward/backward sanity demonstration.

### `train_lm` (tiny decoder-only transformer training demo)

```bash
./build/train_lm
```

This runs a small synthetic next-token training loop and prints loss values. The expected behavior is that loss decreases over training steps.

## 4. Running tests

```bash
cd build
ctest --output-on-failure
```

Or directly:

```bash
./build/tests
```

## 5. Coverage run (optional)

```bash
cmake -S . -B build-coverage -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-coverage
cd build-coverage
ctest --output-on-failure
```

Then use `gcov` / `lcov` as described in `README.md` and `docs/testing.md`.

## 6. Input format notes

- The transformer APIs expect tensor-shaped input data:
  - token ids: `[batch, time]`
  - model activations: `[batch, time, d_model]`
  - logits: `[batch, time, vocab_size]`
- In training demos, token ids are currently represented as float values in `Tensor` and cast to integer indices at usage sites.
