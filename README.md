# vadugrad — MNIST MLP with butterfly layers (Algorithms and AI project)

This repository contains a from-scratch C++ implementation of a small feed-forward network for MNIST, including manual backpropagation. The [specification](docs/specification-doc.md) describes scope, complexity, and sources.

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

## Test coverage (gcov / lcov)

The course expects unit tests with **coverage tracked** (see [Unit testing — coverage](https://algolabra-hy.github.io/unittest-en#has-enough-testing-been-done-test-coverage)). For this C++ project, configure with coverage flags, run tests, then inspect `.gcno`/`.gcda` with `gcov` or generate an HTML report with `lcov`/`genhtml`:

```bash
cmake -S . -B build-coverage -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-coverage
cd build-coverage && ctest --output-on-failure
gcov -b CMakeFiles/vadugrad.dir/src/tensor.cpp.gcno  # example: branch coverage for tensor.cpp
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
| Weekly reports | [weekly-reports/](weekly-reports/) |

Course documentation requirements: [Documentation](https://algolabra-hy.github.io/documentation-en), [Suggested schedule](https://algolabra-hy.github.io/schedule-en).
