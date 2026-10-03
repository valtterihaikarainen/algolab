# User Guide

## Peer review quick path

If you are reviewing this repository, start with the root **`README.md`** section **For peer reviewers**: it lists the exact build and test commands, optional demos, MNIST file names, and a code map. This guide repeats executable details and flags below.

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

### `vadugrad butterfly` (butterfly linear demo)

```bash
./build/vadugrad butterfly
```

This prints a small butterfly linear map sanity check:

- builds an explicit dense matrix by applying the butterfly map to each basis vector
- compares `butterfly_forward(x)` to `x @ dense_matrix^T` (transpose matches the row-vector convention used elsewhere in the project)
- prints one backward statistic (`dL/dx` at index 0) and an example weight gradient block

### `train_lm` (tiny decoder-only transformer training demo)

```bash
./build/train_lm
```

This runs a small synthetic next-token training loop and prints loss values. The expected behavior is that loss decreases over training steps.

### `train_mnist` (MNIST MLP trainer: dense vs butterfly hidden map)

```bash
./build/train_mnist --help
```

**Data layout:** the directory passed to `--data` (default: `data/mnist`) must contain these **exact** training IDX filenames (standard MNIST distribution):

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`

The unit test suite does **not** require these files; see `tests/test_mnist_loader.cpp` for a synthetic minimal IDX fixture.

**Variants:** the hidden block after the first ReLU uses `ButterflyLinear` by default (`n=784`). Pass `--dense` to use an identity hidden map (same tensor shape, no butterfly parameters). `--butterfly` is the default and may be given explicitly for clarity.

Examples (after placing MNIST files under `data/mnist/`):

```bash
./build/train_mnist --data data/mnist --epochs 1 --batch 64 --lr 1e-3
./build/train_mnist --data data/mnist --dense
./build/train_mnist --data data/mnist --butterfly
```

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
