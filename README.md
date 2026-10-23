# Butterfly vs dense layers on MNIST, from scratch in C++

This project asks whether a **butterfly-structured linear layer** can replace a
**dense linear layer** in a small neural network without a meaningful loss in accuracy,
while using far fewer parameters and arithmetic operations. Everything — tensors,
forward and backward passes, and the optimizer — is implemented from scratch in C++17,
without machine-learning libraries or automatic differentiation.

## What is a butterfly layer?

A dense layer multiplies its input by a full *n × n* weight matrix, so it costs *n²*
weights and *n²* operations. The Fast Fourier Transform shows that some *n × n* matrices
can be applied in *O(n log n)* time by splitting the work into log₂ *n* stages that each
mix only **pairs** of positions. A butterfly layer keeps this FFT wiring but makes the
2×2 mixing operations learnable:

| Width *n* | Dense weights (*n²*) | Butterfly weights (2*n* log₂ *n*) |
|-----------|----------------------|-----------------------------------|
| 64        | 4,096                | 768                               |
| 512       | 262,144              | 9,216                             |
| 4,096     | 16,777,216           | 98,304                            |

The catch is that a butterfly layer can only express structured matrices. Whether that is
enough for a real task is what this project measures. The
[specification](docs/specification-doc.md) explains the background in more detail.

## Experiment and results

Three networks (784 → 512 → ReLU → *hidden* → ReLU → 10) are trained on MNIST with
identical settings and differ only in the hidden 512 → 512 layer: `none` (skipped),
`dense`, or `butterfly`. Test accuracy on the 10,000 MNIST test images after 5 epochs,
mean ± standard deviation over 3 seeds, on Linux (GCC) and macOS (Clang):

| Variant     | Hidden-layer parameters | Test accuracy, Linux | Test accuracy, macOS | Seconds per epoch, Linux / macOS |
|-------------|-------------------------|----------------------|----------------------|----------------------------------|
| `none`      | 0                       | 97.73 % ± 0.40       | 97.99 % ± 0.12       | 9.7 / 7.5                        |
| `dense`     | 262,656                 | 97.82 % ± 0.24       | 97.91 % ± 0.03       | 16.0 / 12.5                      |
| `butterfly` | 9,728                   | 97.46 % ± 0.38       | 97.74 % ± 0.22       | 11.8 / 11.9                      |

The butterfly layer comes within 0.4 percentage points of the dense layer with 27× fewer
parameters. The `none` baseline is just as accurate, though, so MNIST is too easy to show
whether the extra hidden layer is needed at all. The butterfly's 28× fewer operations also
translate into only a small speedup, because its memory access pattern is much less
efficient than the dense `matmul`. The
[implementation document](docs/implementation.md#3-results) discusses both.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/tests                                   # unit and integration tests
scripts/download_mnist.sh                       # fetches MNIST into data/mnist
./build/train_mnist --variant butterfly --epochs 3
```

See the [user guide](docs/user-guide.md) for all options.

## Documentation

- [Specification](docs/specification-doc.md): goal, background, planned algorithms and complexity
- [Implementation](docs/implementation.md): program structure, measured results, limitations
- [Testing](docs/testing.md): what is tested, why, coverage
- [User guide](docs/user-guide.md): building, running and inputs
- Weekly reports: [1](weekly-reports/week1.md), [2](weekly-reports/week2.md),
  [3](weekly-reports/week3.md), [4](weekly-reports/week4.md),
  [5](weekly-reports/week5.md), [6](weekly-reports/week6.md), [7](weekly-reports/final-submission.md)
