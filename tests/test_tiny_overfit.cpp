#include <gtest/gtest.h>
#include <vadugrad/nn/decoder_only_transformer.hpp>
#include <vadugrad/nn/nn_ops.hpp>
#include <vadugrad/optim/adam.hpp>

TEST(IntegrationTest, TinyBatchOverfitLossDrops) {
    DecoderOnlyTransformer model(10, 8, 2, 16, 1, 4);
    model.initialize_parameters(0.02f, 7u);
    Adam optimizer(2e-2f);

    Tensor x({1, 4});
    x({0, 0}) = 1.0f;
    x({0, 1}) = 2.0f;
    x({0, 2}) = 3.0f;
    x({0, 3}) = 4.0f;

    Tensor y({1, 4});
    y({0, 0}) = 2.0f;
    y({0, 1}) = 3.0f;
    y({0, 2}) = 4.0f;
    y({0, 3}) = 5.0f;

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int step = 0; step < 80; ++step) {
        const Tensor logits = model.forward(x);
        const float loss = cross_entropy_mean(logits, y);
        const Tensor grad_logits = cross_entropy_grad_logits(logits, y);
        const auto grads = model.backward(grad_logits);
        model.apply_gradients(grads, optimizer);
        if (step == 0) {
            first_loss = loss;
        }
        last_loss = loss;
    }

    EXPECT_LT(last_loss, first_loss);
}
