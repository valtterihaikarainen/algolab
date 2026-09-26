#include <iostream>
#include <vadugrad/nn/decoder_only_transformer.hpp>
#include <vadugrad/nn/nn_ops.hpp>
#include <vadugrad/optim/adam.hpp>

int main() {
    constexpr int vocab = 12;
    constexpr int d_model = 16;
    constexpr int num_heads = 4;
    constexpr int d_ff = 32;
    constexpr int layers = 2;
    constexpr int seq = 6;
    constexpr int batch = 2;

    DecoderOnlyTransformer model(vocab, d_model, num_heads, d_ff, layers, seq);
    model.initialize_parameters(0.02f, 123u);
    Adam optimizer(1e-2f);

    Tensor x({batch, seq});
    Tensor y({batch, seq});
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            x({b, t}) = static_cast<float>((b + t) % vocab);
            y({b, t}) = static_cast<float>((b + t + 1) % vocab);
        }
    }

    for (int step = 0; step < 100; ++step) {
        const Tensor logits = model.forward(x);
        const float loss = cross_entropy_mean(logits, y);
        const Tensor grad_logits = cross_entropy_grad_logits(logits, y);
        const auto grads = model.backward(grad_logits);
        model.apply_gradients(grads, optimizer);

        if (step % 10 == 0 || step == 99) {
            std::cout << "step=" << step << " loss=" << loss << "\n";
        }
    }
    return 0;
}
