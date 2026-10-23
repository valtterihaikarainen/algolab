#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vadugrad/nn/mlp.hpp>
#include <vadugrad/nn/nn_ops.hpp>

#include "grad_check.hpp"

namespace {

const HiddenVariant kAllVariants[] = {HiddenVariant::None, HiddenVariant::Dense,
                                      HiddenVariant::Butterfly};

std::string name_of(HiddenVariant variant) {
    switch (variant) {
        case HiddenVariant::None: return "none";
        case HiddenVariant::Dense: return "dense";
        case HiddenVariant::Butterfly: return "butterfly";
    }
    return "?";
}

/**
 * Four Gaussian clusters in 16 dimensions. Cluster c is centred at +4 on coordinates 4c..4c+3
 * and 0 elsewhere, with unit noise, so the classes are well separated but not trivially so.
 */
struct Clusters {
    Tensor points;
    Tensor labels;
};

Clusters make_clusters(int per_class, unsigned int seed) {
    constexpr int kClasses = 4;
    constexpr int kDim = 16;
    Clusters data{grad_check::random_tensor({kClasses * per_class, kDim}, seed),
                  Tensor({kClasses * per_class, 1})};
    for (int row = 0; row < kClasses * per_class; ++row) {
        const int label = row % kClasses;
        data.labels(row, 0) = static_cast<float>(label);
        for (int d = 4 * label; d < 4 * label + 4; ++d) {
            data.points(row, d) += 4.0f;
        }
    }
    return data;
}

double accuracy(const Tensor& logits, const Tensor& labels) {
    int correct = 0;
    for (int b = 0; b < logits.shape()[0]; ++b) {
        int best = 0;
        for (int c = 1; c < logits.shape()[1]; ++c) {
            if (logits(b, c) > logits(b, best)) best = c;
        }
        if (best == static_cast<int>(labels(b, 0))) ++correct;
    }
    return static_cast<double>(correct) / logits.shape()[0];
}

}  // namespace

// The parameter counts reported in the experiment (784 → 512 → hidden → 10).
TEST(MlpTest, ParameterCountsAtExperimentSize) {
    const int shared = 784 * 512 + 512 + 512 * 10 + 10;
    EXPECT_EQ(Mlp(784, 512, 10, HiddenVariant::None).parameter_count(), shared);
    EXPECT_EQ(Mlp(784, 512, 10, HiddenVariant::Dense).hidden_parameter_count(), 512 * 512 + 512);
    EXPECT_EQ(Mlp(784, 512, 10, HiddenVariant::Butterfly).hidden_parameter_count(), 9 * 256 * 4 + 512);
    EXPECT_EQ(Mlp(784, 512, 10, HiddenVariant::Butterfly).parameter_count(), shared + 9728);
}

TEST(MlpTest, ButterflyNeedsPowerOfTwoWidth) {
    EXPECT_THROW(Mlp(6, 12, 3, HiddenVariant::Butterfly), std::invalid_argument);
    EXPECT_NO_THROW(Mlp(6, 12, 3, HiddenVariant::Dense));
}

TEST(MlpTest, ApplyGradientsBeforeBackwardThrows) {
    Mlp model(4, 4, 2, HiddenVariant::Dense);
    Adam optimizer;
    EXPECT_THROW(model.apply_gradients(optimizer), std::logic_error);
}

// The experiment compares variants trained from the same seed, so a seed must fully
// determine the network.
TEST(MlpTest, SameSeedGivesIdenticalNetworks) {
    const Tensor x = grad_check::random_tensor({3, 8}, 71u);
    for (HiddenVariant variant : kAllVariants) {
        Mlp first(8, 8, 3, variant);
        Mlp second(8, 8, 3, variant);
        first.initialize(5u);
        second.initialize(5u);
        const Tensor a = first.forward(x);
        const Tensor b = second.forward(x);
        for (int i = 0; i < a.numel(); ++i) {
            ASSERT_EQ(a.data()[i], b.data()[i]) << name_of(variant);
        }
    }
}

// End-to-end check of the whole backward pass, including how the layers are chained: the
// gradient of the cross-entropy loss with respect to every parameter of every layer is
// compared against finite differences.
TEST(MlpTest, EndToEndGradientsMatchFiniteDifferences) {
    const Tensor x = grad_check::random_tensor({4, 6}, 81u);
    Tensor labels({4, 1});
    for (int b = 0; b < 4; ++b) labels(b, 0) = static_cast<float>(b % 3);

    for (HiddenVariant variant : kAllVariants) {
        Mlp model(6, 8, 3, variant);
        model.initialize(9u);
        const Tensor logits = model.forward(x);
        model.backward(cross_entropy_grad_logits_rows(logits, labels));
        const std::vector<Tensor> analytic = model.gradients();

        auto loss = [&] { return static_cast<double>(cross_entropy_mean_rows(model.forward(x), labels)); };
        const std::vector<Tensor*> params = model.parameters();
        ASSERT_EQ(params.size(), analytic.size());
        for (std::size_t p = 0; p < params.size(); ++p) {
            grad_check::expect_close(analytic[p], grad_check::numerical_gradient(*params[p], loss, 1e-3f),
                                     2e-2f, name_of(variant) + " parameter " + std::to_string(p));
        }
    }
}

// Integration test of the full training loop (forward, loss, backward, Adam) on data with a
// known answer: every variant must learn the four clusters and generalize to fresh samples.
TEST(MlpTest, EveryVariantLearnsSeparableClusters) {
    const Clusters train = make_clusters(50, 91u);
    const Clusters held_out = make_clusters(25, 92u);

    for (HiddenVariant variant : kAllVariants) {
        Mlp model(16, 16, 4, variant);
        model.initialize(3u);
        Adam optimizer(1e-2f);
        const float initial_loss = cross_entropy_mean_rows(model.forward(train.points), train.labels);
        for (int step = 0; step < 150; ++step) {
            const Tensor logits = model.forward(train.points);
            model.backward(cross_entropy_grad_logits_rows(logits, train.labels));
            model.apply_gradients(optimizer);
        }
        const Tensor train_logits = model.forward(train.points);
        EXPECT_LT(cross_entropy_mean_rows(train_logits, train.labels), 0.1f * initial_loss)
            << name_of(variant);
        EXPECT_GE(accuracy(train_logits, train.labels), 0.98) << name_of(variant);
        EXPECT_GE(accuracy(model.forward(held_out.points), held_out.labels), 0.95) << name_of(variant);
    }
}
