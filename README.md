# vadugrad — from-scratch C++ neural nets and transformer blocks

This repository contains from-scratch C++ implementations of tensor ops, manual backprop modules, and a decoder-only transformer training demo. The [specification](docs/specification-doc.md) describes original scope, complexity, and sources.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

The first configure downloads [GoogleTest](https://github.com/google/googletest) via CMake `FetchContent` (network required).

## Run tests

```bash
cd build && ctest --output-on-failure
```

Or run the test binary directly: `./build/tests`.

## Run the CLI demo

After building, the executable is `./build/vadugrad` (target name `vadugrad_cli`). It prints a small dense-layer forward pass and one backward step. Use `./build/vadugrad --help` for options.

## Run the tiny transformer training demo

After building, run:

```bash
./build/train_lm
```

This runs a tiny synthetic next-token training loop and prints loss values.

## Test coverage (gcov / lcov)

The course expects unit tests with **coverage tracked** (see [Unit testing — coverage](https://algolabra-hy.github.io/unittest-en#has-enough-testing-been-done-test-coverage)). For this C++ project, configure with coverage flags, run tests, then inspect `.gcno`/`.gcda` with `gcov` or generate an HTML report with `lcov`/`genhtml`:

```bash
cmake -S . -B build-coverage -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-coverage
cd build-coverage && ctest --output-on-failure
gcov -b -s ../src CMakeFiles/vadugrad.dir/src/tensor.cpp.gcno CMakeFiles/vadugrad.dir/src/dense_linear.cpp.gcno
```

With `lcov` installed:

```bash
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/_deps/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

Open `coverage_html/index.html` in a browser.

## Documentation in the repository

| Document | Path |
|----------|------|
| Specification | [docs/specification-doc.md](docs/specification-doc.md) |
| Implementation | [docs/implementation.md](docs/implementation.md) |
| Testing | [docs/testing.md](docs/testing.md) |
| User guide | [docs/user-guide.md](docs/user-guide.md) |
| Weekly reports | [week 1](weekly-reports/week1.md), [week 2](weekly-reports/week2.md), [week 3](weekly-reports/week3.md), [week 4](weekly-reports/week4.md) |

Course documentation requirements: [Documentation](https://algolabra-hy.github.io/documentation-en), [Suggested schedule](https://algolabra-hy.github.io/schedule-en).
