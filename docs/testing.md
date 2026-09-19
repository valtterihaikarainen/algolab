# Testing

This document describes how **vadugrad** is unit tested and how to collect **line and branch coverage** for the C++ sources (course analogue to Python `coverage`).

## Framework and layout

- **Framework**: [GoogleTest](https://github.com/google/googletest), pulled by CMake `FetchContent` when configuring the project.
- **Test sources**: `tests/test_tensor.cpp` (tensor utilities, `matmul`, `sum`, etc.) and `tests/test_dense_linear.cpp` (dense linear layer forward/backward with hand-checked values and batching).

## Running tests

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cd build && ctest --output-on-failure
```

Or run the test binary directly: `./build/tests`.

## What is covered

Representative checks include:

- Row-major strides, tensor copy/move behaviour, `fill`, `reshape` invariants, out-of-range indexing.
- Elementwise ops, `transpose2d`, `matmul`, `sum` along an axis (with and without `keepdim`).
- **Dense linear layer**: forward @f$y = xW + b@f$, backward gradients @f$\partial L/\partial x@f$, @f$\partial L/\partial W@f$, @f$\partial L/\partial b@f$ for single-sample and batched inputs, plus shape/validation errors.

## Coverage report (gcov)

Configure with `ENABLE_COVERAGE=ON` (see the main [README](../README.md)), build, run tests, then generate `.gcov` files from the build directory:

```bash
cd build
gcov -b -s ../src CMakeFiles/vadugrad.dir/src/tensor.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/dense_linear.cpp.gcno
```

This writes `tensor.cpp.gcov` and `dense_linear.cpp.gcov` under `build/`. Open them in an editor or use `lcov`/`genhtml` as described in the README for an HTML summary.

### Snapshot (week 3, current suite)

The following figures are **illustrative**: they depend on the exact compiler, flags, and tests. Regenerate after changing code or tests.

| Translation unit   | Line coverage (gcov) |
|--------------------|------------------------|
| `src/tensor.cpp`   | about 88%              |
| `src/dense_linear.cpp` | about 83%          |

`src/main.cpp` is not linked into the test binary; it is exercised by running `./build/vadugrad` manually.

## Static analysis (C++)

The course text may mention **pylint** for Python projects. For this C++ codebase, comparable hygiene is:

- Compiler warnings: `-Wall -Wextra` (enabled in `CMakeLists.txt`).
- Optional: `clang-tidy` or `cppcheck` on `src/` and `include/` (not wired into CMake by default).
