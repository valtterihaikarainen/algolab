# vadugrad — from-scratch C++ neural nets and transformer blocks

This repository contains from-scratch C++ implementations of tensor ops, manual backprop modules, and a decoder-only transformer training demo. The [specification](docs/specification-doc.md) describes original scope, complexity, and sources.

## For peer reviewers

Use this section as a single entry point; the rest of the README and linked docs fill in detail.

### What to run (in order)

1. **Configure and build** (first configure needs network for GoogleTest via CMake `FetchContent`):

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build
   ```

2. **Run the full test suite** (no MNIST download required; the MNIST loader is covered with a tiny synthetic IDX fixture in `tests/test_mnist_loader.cpp`):

   ```bash
   cd build && ctest --output-on-failure
   ```

   Equivalent: `./build/tests` from the repo root after a successful build.

3. **Optional demos** (manual inspection, not required for CI-style verification):

   | Command | Purpose |
   |---------|---------|
   | `./build/vadugrad` | Small dense `DenseLinear` forward/backward sanity demo |
   | `./build/vadugrad butterfly` | `ButterflyLinear`: dense-matrix equivalence check + sample backward stats |
   | `./build/train_lm` | Tiny synthetic next-token training (decoder-only stack) |
   | `./build/train_mnist --help` | MNIST MLP trainer CLI (`--dense` vs `--butterfly` hidden; see below) |

### MNIST files (only if you run `train_mnist`)

Place the official **training** IDX files in a directory (e.g. `data/mnist/`) and pass `--data` to that directory. Expected **exact** filenames:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`

These are the standard files from the [MNIST dataset page](http://yann.lecun.com/exdb/mnist/) (or mirrors). If they are missing, `train_mnist` fails at load time with a clear open/read error.

Example:

```bash
./build/train_mnist --data data/mnist --epochs 1 --batch 64 --lr 1e-3
./build/train_mnist --data data/mnist --dense    # skip butterfly in the hidden block
./build/train_mnist --data data/mnist --butterfly # default; explicit for reviewers
```

### Where to look in the code (high signal)

| Topic | Primary locations |
|-------|-------------------|
| Butterfly layer | `include/vadugrad/butterfly_linear.hpp`, `src/butterfly_linear.cpp`, `tests/test_butterfly_linear.cpp`, CLI in `src/main.cpp` (`butterfly` subcommand) |
| MNIST loading | `include/vadugrad/data/mnist.hpp`, `src/mnist.cpp`, `tests/test_mnist_loader.cpp` |
| MNIST MLP training app | `apps/train_mnist.cpp` |
| Row softmax / CE for batched logits | `include/vadugrad/nn/nn_ops.hpp`, `src/nn_ops.cpp`, `tests/test_nn_ops.cpp` |

### Documentation map (course / review)

| Document | Role |
|----------|------|
| [docs/specification-doc.md](docs/specification-doc.md) | Scope, complexity, sources |
| [docs/implementation.md](docs/implementation.md) | Module layout, complexity notes, limitations |
| [docs/testing.md](docs/testing.md) | What each test area covers, coverage commands |
| [docs/user-guide.md](docs/user-guide.md) | Requirements, all executables, flags |
| [weekly-reports/](weekly-reports/) | Weekly progress notes |

Course pointers: [Documentation](https://algolabra-hy.github.io/documentation-en), [Unit testing — coverage](https://algolabra-hy.github.io/unittest-en#has-enough-testing-been-done-test-coverage).

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

After building, the executable is `./build/vadugrad` (target name `vadugrad_cli`).

- Default: a small dense-layer forward pass and one backward step.
- Butterfly demo: `./build/vadugrad butterfly`

Use `./build/vadugrad --help` for options.

## Run the tiny transformer training demo

After building, run:

```bash
./build/train_lm
```

This runs a tiny synthetic next-token training loop and prints loss values.

## Run MNIST training (MLP)

After building, see **For peer reviewers** above for required filenames and `--data`. Quick start:

```bash
./build/train_mnist --help
./build/train_mnist --data data/mnist --epochs 1 --batch 64 --lr 1e-3
```

## Test coverage (gcov / lcov)

The course expects unit tests with **coverage tracked** (see [Unit testing — coverage](https://algolabra-hy.github.io/unittest-en#has-enough-testing-been-done-test-coverage)). For this C++ project, configure with coverage flags, run tests, then inspect `.gcno`/`.gcda` with `gcov` or generate an HTML report with `lcov`/`genhtml`:

```bash
cmake -S . -B build-coverage -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-coverage
cd build-coverage && ctest --output-on-failure
gcov -b -s ../src \
  CMakeFiles/vadugrad.dir/src/tensor.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/dense_linear.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/butterfly_linear.cpp.gcno \
  CMakeFiles/vadugrad.dir/src/mnist.cpp.gcno
```

With `lcov` installed:

```bash
lcov --capture --directory . --output-file coverage.info --ignore-errors inconsistent,inconsistent
lcov --remove coverage.info '/usr/*' '*/_deps/*' --output-file coverage.info --ignore-errors inconsistent,inconsistent
genhtml coverage.info --output-directory coverage_html
```

Open `coverage_html/index.html` in a browser.

## Documentation in the repository

Peer reviewers: the table in **For peer reviewers** lists the same paths with short roles. Quick links:

| Document | Path |
|----------|------|
| Specification | [docs/specification-doc.md](docs/specification-doc.md) |
| Implementation | [docs/implementation.md](docs/implementation.md) |
| Testing | [docs/testing.md](docs/testing.md) |
| User guide | [docs/user-guide.md](docs/user-guide.md) |
| Weekly reports | [week 1](weekly-reports/week1.md), [week 2](weekly-reports/week2.md), [week 3](weekly-reports/week3.md), [week 4](weekly-reports/week4.md), [week 5](weekly-reports/week5.md) |

Course documentation requirements: [Documentation](https://algolabra-hy.github.io/documentation-en), [Suggested schedule](https://algolabra-hy.github.io/schedule-en).
