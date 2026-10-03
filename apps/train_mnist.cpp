#include <vadugrad/butterfly_linear.hpp>
#include <vadugrad/data/mnist.hpp>
#include <vadugrad/dense_linear.hpp>
#include <vadugrad/nn/activations.hpp>
#include <vadugrad/nn/nn_ops.hpp>
#include <vadugrad/nn/parameter_init.hpp>
#include <vadugrad/optim/adam.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string data_dir = "data/mnist";
    int epochs = 1;
    int batch = 64;
    float lr = 1e-3f;
    bool use_butterfly = true;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--data" && i + 1 < argc) {
            a.data_dir = argv[++i];
        } else if (s == "--epochs" && i + 1 < argc) {
            a.epochs = std::stoi(argv[++i]);
        } else if (s == "--batch" && i + 1 < argc) {
            a.batch = std::stoi(argv[++i]);
        } else if (s == "--lr" && i + 1 < argc) {
            a.lr = std::stof(argv[++i]);
        } else if (s == "--dense") {
            a.use_butterfly = false;
        } else if (s == "--butterfly") {
            a.use_butterfly = true;
        } else if (s == "-h" || s == "--help") {
            std::cout << "train_mnist — tiny MNIST MLP trainer (manual backprop)\n";
            std::cout << "Usage:\n";
            std::cout << "  train_mnist [--data DIR] [--epochs N] [--batch B] [--lr X] [--dense|--butterfly]\n";
            std::cout << "Defaults:\n";
            std::cout << "  --data data/mnist\n";
            std::cout << "  --epochs 1 --batch 64 --lr 1e-3\n";
            std::cout << "  hidden uses butterfly by default; pass --dense for dense hidden map\n";
            std::cout << "\nExpected MNIST files in DIR:\n";
            std::cout << "  train-images-idx3-ubyte\n";
            std::cout << "  train-labels-idx1-ubyte\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown arg: " + s);
        }
    }
    if (a.epochs <= 0 || a.batch <= 0) {
        throw std::invalid_argument("epochs/batch must be positive");
    }
    return a;
}

Tensor rows_slice(const Tensor& m, int row0, int rows) {
    if (m.ndim() != 2) {
        throw std::invalid_argument("rows_slice expects rank-2 tensor");
    }
    const int cols = m.shape()[1];
    Tensor out({rows, cols});
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out({r, c}) = m({row0 + r, c});
        }
    }
    return out;
}

void init_dense(DenseLinear& layer, unsigned int& seed) {
    initialize_tensor_normal(layer.weight(), 0.02f, seed);
    layer.bias().fill(0.0f);
}

void init_butterfly(ButterflyLinear& layer, unsigned int& seed) {
    for (int s = 0; s < layer.num_stages(); ++s) {
        initialize_tensor_normal(layer.stage_weight(s), 0.02f, seed);
    }
}

float accuracy(const Tensor& logits, const Tensor& labels01) {
    int correct = 0;
    const int bsz = logits.shape()[0];
    for (int b = 0; b < bsz; ++b) {
        int argmax = 0;
        float best = logits({b, 0});
        for (int c = 1; c < logits.shape()[1]; ++c) {
            if (logits({b, c}) > best) {
                best = logits({b, c});
                argmax = c;
            }
        }
        const int y = static_cast<int>(labels01({b, 0}));
        if (argmax == y) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(bsz);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    const MnistDataset data = load_mnist_train(args.data_dir);
    const int n = data.images.shape()[0];

    constexpr int in_dim = 784;
    constexpr int hid = 512;  // power of two for ButterflyLinear
    constexpr int out_dim = 10;

    DenseLinear l1(in_dim, hid);
    DenseLinear l2(hid, out_dim);
    ButterflyLinear bf(hid);

    unsigned int seed = 123u;
    init_dense(l1, seed);
    init_dense(l2, seed);
    init_butterfly(bf, seed);

    Adam opt(args.lr);

    std::mt19937 rng(7);
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<std::size_t>(i)] = i;
    }

    for (int e = 0; e < args.epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double loss_acc = 0.0;
        int steps = 0;

        for (int start = 0; start < n; start += args.batch) {
            const int bsz = std::min(args.batch, n - start);
            Tensor xb({bsz, data.images.shape()[1]});
            Tensor yb({bsz, 1});
            for (int r = 0; r < bsz; ++r) {
                const int row = idx[static_cast<std::size_t>(start + r)];
                for (int c = 0; c < data.images.shape()[1]; ++c) {
                    xb({r, c}) = data.images({row, c});
                }
                yb({r, 0}) = data.labels({row, 0});
            }

            const Tensor h1_pre = l1.forward(xb);
            const Tensor h1 = relu(h1_pre);

            const Tensor h2 = args.use_butterfly ? bf.forward(h1) : h1;

            const Tensor logits = l2.forward(h2);
            const float loss = cross_entropy_mean_rows(logits, yb);
            loss_acc += static_cast<double>(loss);
            ++steps;

            const Tensor glog = cross_entropy_grad_logits_rows(logits, yb);
            const auto g2 = l2.backward(h2, glog);
            Tensor grad_h2 = g2.grad_input;

            Tensor grad_h1 = grad_h2;
            if (args.use_butterfly) {
                const auto gb = bf.backward(grad_h2);
                grad_h1 = gb.grad_input;
                for (int s = 0; s < bf.num_stages(); ++s) {
                    opt.step(bf.stage_weight(s), gb.grad_stages[static_cast<std::size_t>(s)]);
                }
            }

            const Tensor grad_h1_pre = relu_backward(h1_pre, grad_h1);
            const auto g1 = l1.backward(xb, grad_h1_pre);

            opt.step(l1.weight(), g1.grad_weight);
            opt.step(l1.bias(), g1.grad_bias);
            opt.step(l2.weight(), g2.grad_weight);
            opt.step(l2.bias(), g2.grad_bias);
        }

        std::cout << "epoch=" << e + 1 << " mean_loss=" << (loss_acc / std::max(1, steps)) << "\n";
    }

    // quick train accuracy on a few batches (same as training order isn't important here)
    {
        int checks = std::min(n, 5 * args.batch);
        float acc_sum = 0.0f;
        int acc_steps = 0;
        for (int start = 0; start < checks; start += args.batch) {
            const int bsz = std::min(args.batch, checks - start);
            Tensor xb = rows_slice(data.images, start, bsz);
            Tensor yb = rows_slice(data.labels, start, bsz);
            const Tensor h1_pre = l1.forward(xb);
            const Tensor h1 = relu(h1_pre);
            const Tensor h2 = args.use_butterfly ? bf.forward(h1) : h1;
            const Tensor logits = l2.forward(h2);
            acc_sum += accuracy(logits, yb);
            ++acc_steps;
        }
        std::cout << "approx_train_acc(first " << checks << " examples, mean over "
                  << acc_steps << " batches)=" << (acc_sum / static_cast<float>(acc_steps)) << "\n";
    }

    std::cout << "variant=" << (args.use_butterfly ? "butterfly_hidden" : "dense_hidden") << "\n";
    return 0;
}
