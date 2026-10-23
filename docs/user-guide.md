# User Guide

## 1. Requirements

- A C++17 compiler (GCC or Clang) and CMake 3.20 or newer
- `curl` and `gunzip` for downloading MNIST
- Network access for the first CMake configure (GoogleTest is downloaded automatically)
- Optional: Python 3 with matplotlib for the results plot, `gcovr` for the coverage report

Tested on Linux (GCC, aarch64) and macOS (Apple Clang, Apple Silicon). On macOS, CMake can be
installed with `brew install cmake`; everything else comes with the Xcode command-line tools.

If GoogleTest cannot be downloaded (for example offline), point CMake at a local copy:
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

## 2. Building

Use a **Release** build for training: it is compiled with optimizations and is many times
faster than a Debug build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces two executables: `build/train_mnist` (the experiment) and `build/tests` (the
test suite).

## 3. Getting the MNIST data

```bash
scripts/download_mnist.sh            # downloads into data/mnist
scripts/download_mnist.sh other/dir  # or into another directory
```

The script downloads and unpacks the four standard MNIST files. It skips files that are
already present, so it is safe to run again. The resulting directory contains:

| File                        | Contents                     |
|-----------------------------|------------------------------|
| `train-images-idx3-ubyte`   | 60,000 training images       |
| `train-labels-idx1-ubyte`   | their labels (digits 0–9)    |
| `t10k-images-idx3-ubyte`    | 10,000 test images           |
| `t10k-labels-idx1-ubyte`    | their labels                 |

`data/` is listed in `.gitignore`, so the dataset is never committed.

## 4. Training a network

```bash
./build/train_mnist --variant butterfly --epochs 5
```

| Option                          | Meaning                                              | Default      |
|---------------------------------|------------------------------------------------------|--------------|
| `--variant none\|dense\|butterfly` | Type of the hidden 512 → 512 layer                | `butterfly`  |
| `--epochs N`                    | Passes over the 60,000 training images               | 3            |
| `--batch B`                     | Mini-batch size                                      | 64           |
| `--lr X`                        | Adam learning rate                                   | 0.001        |
| `--seed S`                      | Positive integer; fixes initial weights and data order | 1          |
| `--data DIR`                    | Directory containing the MNIST files                 | `data/mnist` |
| `--log FILE`                    | Also write one CSV row per epoch to `FILE`           | off          |
| `--help`                        | Print the options                                    |              |

Example output:

```text
variant=butterfly seed=1 hidden_params=9728 total_params=416778
epoch=1 train_loss=0.2533 test_loss=0.1076 test_accuracy=0.9679 epoch_seconds=11.5759
...
```

- `hidden_params`: learnable values in the hidden layer only (0, 262,656 or 9,728).
- `train_loss`: mean cross-entropy over the epoch's training batches.
- `test_loss`, `test_accuracy`: measured on all 10,000 test images after the epoch.
  Accuracy is a fraction: 0.9679 means 96.79 % correct.
- `epoch_seconds`: wall-clock time of the training pass (evaluation excluded).

Running the same command twice gives the same numbers, because `--seed` fixes both the
initial weights and the order of the training data. This holds for one compiler and standard
library: the data order comes from `std::shuffle`, whose algorithm differs between GCC's
libstdc++ and Clang's libc++, so Linux and macOS runs with the same seed give slightly
different results. The program exits with an error message
for unknown options, missing values, a non-power-of-two butterfly width, or missing data
files.

## 5. Reproducing the experiment

```bash
mkdir -p results
for seed in 1 2 3; do
  for variant in none dense butterfly; do
    ./build/train_mnist --variant $variant --seed $seed --epochs 5 \
      --log results/${variant}_seed${seed}.csv
  done
done
python3 scripts/plot_results.py results docs/img/test_curves.png
```

`plot_results.py` prints the results table (mean and standard deviation over seeds) and
saves the test-loss and test-accuracy plot. Run the variants one at a time, as above, so
that the timings are not distorted by runs competing for the CPU.

## 6. Running the tests

```bash
./build/tests          # or: cd build && ctest --output-on-failure
```

The tests use synthetic data and do not need the MNIST download. For a coverage report, see
the [testing document](testing.md#5-reproducing-the-tests-and-coverage).
