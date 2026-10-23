/**
 * @file train_mnist.cpp
 * @brief Command-line program that trains one network variant on MNIST and reports its test
 *        accuracy, parameter counts and training time.
 */

#include <vadugrad/data/mnist.hpp>
#include <vadugrad/nn/mlp.hpp>
#include <vadugrad/nn/nn_ops.hpp>
#include <vadugrad/optim/adam.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputDim = 784;
constexpr int kHiddenDim = 512;
constexpr int kNumClasses = 10;
constexpr int kEvalBatch = 1000;

/** Command-line settings; the defaults are the values used in the experiment. */
struct Options {
    std::string data_dir = "data/mnist";
    HiddenVariant variant = HiddenVariant::Butterfly;
    int epochs = 3;
    int batch_size = 64;
    float learning_rate = 1e-3f;
    unsigned int seed = 1;
    std::string log_path;
};

void print_usage() {
    std::cout << "Usage: train_mnist [options]\n"
                 "  --variant none|dense|butterfly  hidden layer type (default: butterfly)\n"
                 "  --epochs N                      passes over the training set (default: 3)\n"
                 "  --batch B                       mini-batch size (default: 64)\n"
                 "  --lr X                          Adam learning rate (default: 0.001)\n"
                 "  --seed S                        random seed, positive integer (default: 1)\n"
                 "  --data DIR                      directory with the MNIST IDX files (default: data/mnist)\n"
                 "  --log FILE                      also write per-epoch results as CSV\n"
                 "  --help                          show this message\n";
}

/** Map "none", "dense" or "butterfly" to a @ref HiddenVariant; throws on anything else. */
HiddenVariant parse_variant(const std::string& name) {
    if (name == "none") return HiddenVariant::None;
    if (name == "dense") return HiddenVariant::Dense;
    if (name == "butterfly") return HiddenVariant::Butterfly;
    throw std::invalid_argument("unknown variant '" + name + "' (expected none, dense or butterfly)");
}

std::string variant_name(HiddenVariant variant) {
    switch (variant) {
        case HiddenVariant::None: return "none";
        case HiddenVariant::Dense: return "dense";
        case HiddenVariant::Butterfly: return "butterfly";
    }
    return "unknown";
}

/** Read "--name value" pairs into @ref Options; throws on unknown or incomplete options. */
Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            print_usage();
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "--variant") options.variant = parse_variant(value);
        else if (flag == "--epochs") options.epochs = std::stoi(value);
        else if (flag == "--batch") options.batch_size = std::stoi(value);
        else if (flag == "--lr") options.learning_rate = std::stof(value);
        else if (flag == "--seed") options.seed = static_cast<unsigned int>(std::stoul(value));
        else if (flag == "--data") options.data_dir = value;
        else if (flag == "--log") options.log_path = value;
        else throw std::invalid_argument("unknown option " + flag);
    }
    if (options.epochs <= 0 || options.batch_size <= 0 || options.learning_rate <= 0.0f ||
        options.seed == 0) {
        throw std::invalid_argument("epochs, batch, lr and seed must be positive");
    }
    return options;
}

/** Copy the rows listed in order[start, start + count) into a new batch. */
MnistDataset make_batch(const MnistDataset& data, const std::vector<int>& order, int start, int count) {
    const int width = data.images.shape()[1];
    MnistDataset batch{Tensor({count, width}), Tensor({count, 1})};
    for (int r = 0; r < count; ++r) {
        const int source_row = order[static_cast<std::size_t>(start + r)];
        std::copy_n(data.images.data() + source_row * width, width, batch.images.data() + r * width);
        batch.labels.data()[r] = data.labels.data()[source_row];
    }
    return batch;
}

/** @return How many rows of @p logits have their highest score at the correct label. */
int count_correct(const Tensor& logits, const Tensor& labels) {
    int correct = 0;
    const int num_classes = logits.shape()[1];
    for (int b = 0; b < logits.shape()[0]; ++b) {
        const float* row = logits.data() + b * num_classes;
        const int predicted = static_cast<int>(std::max_element(row, row + num_classes) - row);
        if (predicted == static_cast<int>(labels.data()[b])) {
            ++correct;
        }
    }
    return correct;
}

/** One pass over the shuffled training set. @return Mean training loss. */
double train_epoch(Mlp& model, Adam& optimizer, const MnistDataset& train, std::mt19937& rng,
                   int batch_size) {
    std::vector<int> order(static_cast<std::size_t>(train.images.shape()[0]));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    double loss_sum = 0.0;
    int num_batches = 0;
    for (int start = 0; start < static_cast<int>(order.size()); start += batch_size) {
        const int count = std::min(batch_size, static_cast<int>(order.size()) - start);
        const MnistDataset batch = make_batch(train, order, start, count);
        const Tensor logits = model.forward(batch.images);
        loss_sum += cross_entropy_mean_rows(logits, batch.labels);
        model.backward(cross_entropy_grad_logits_rows(logits, batch.labels));
        model.apply_gradients(optimizer);
        ++num_batches;
    }
    return loss_sum / num_batches;
}

struct Evaluation {
    double loss;
    double accuracy;
};

/** Mean loss and accuracy on @p data, in batches of kEvalBatch, without updating weights. */
Evaluation evaluate(Mlp& model, const MnistDataset& data) {
    const int total = data.images.shape()[0];
    std::vector<int> order(static_cast<std::size_t>(total));
    std::iota(order.begin(), order.end(), 0);

    double loss_sum = 0.0;
    int correct = 0;
    for (int start = 0; start < total; start += kEvalBatch) {
        const int count = std::min(kEvalBatch, total - start);
        const MnistDataset batch = make_batch(data, order, start, count);
        const Tensor logits = model.forward(batch.images);
        loss_sum += cross_entropy_mean_rows(logits, batch.labels) * count;
        correct += count_correct(logits, batch.labels);
    }
    return Evaluation{loss_sum / total, static_cast<double>(correct) / total};
}

/** Load the data, train for the requested epochs, print and optionally log each epoch. */
int run(const Options& options) {
    const MnistDataset train = load_mnist(options.data_dir, MnistSplit::Train);
    const MnistDataset test = load_mnist(options.data_dir, MnistSplit::Test);

    Mlp model(kInputDim, kHiddenDim, kNumClasses, options.variant);
    model.initialize(options.seed);
    Adam optimizer(options.learning_rate);
    std::mt19937 rng(options.seed);

    std::cout << "variant=" << variant_name(options.variant) << " seed=" << options.seed
              << " hidden_params=" << model.hidden_parameter_count()
              << " total_params=" << model.parameter_count() << "\n";

    std::ofstream log;
    if (!options.log_path.empty()) {
        log.open(options.log_path);
        if (!log) {
            throw std::runtime_error("could not open log file " + options.log_path);
        }
        log << "variant,seed,epoch,train_loss,test_loss,test_accuracy,epoch_seconds,hidden_params\n";
    }

    std::cout << std::fixed << std::setprecision(4);
    for (int epoch = 1; epoch <= options.epochs; ++epoch) {
        const auto started = std::chrono::steady_clock::now();
        const double train_loss = train_epoch(model, optimizer, train, rng, options.batch_size);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const Evaluation result = evaluate(model, test);

        std::cout << "epoch=" << epoch << " train_loss=" << train_loss << " test_loss=" << result.loss
                  << " test_accuracy=" << result.accuracy << " epoch_seconds=" << seconds << "\n";
        if (log) {
            log << variant_name(options.variant) << ',' << options.seed << ',' << epoch << ','
                << train_loss << ',' << result.loss << ',' << result.accuracy << ',' << seconds << ','
                << model.hidden_parameter_count() << '\n';
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "train_mnist: " << error.what() << "\n";
        return 1;
    }
}
