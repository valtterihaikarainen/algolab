/**
 * @file parameter_init.cpp
 * @brief Reproducible normal random numbers. The public API is documented in
 *        parameter_init.hpp.
 */

#include <vadugrad/nn/parameter_init.hpp>

#include <cmath>
#include <stdexcept>

namespace {

/** Advance @p state and return the next 32-bit pseudo-random number. */
unsigned int next_u32(unsigned int& state) {
    // Marsaglia xorshift32. A zero state would stay zero forever, hence the seed check.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/** @return A uniform sample strictly inside (0, 1); 0 must be excluded because of log(u1) below. */
float uniform_01(unsigned int& state) {
    const unsigned int x = next_u32(state);
    constexpr double kUint32RangePlusOne = 4294967297.0;  // 2^32 + 1, keeps samples inside open interval (0, 1).
    return static_cast<float>((x + 1.0) / kUint32RangePlusOne);
}
}  // namespace

void initialize_tensor_normal(Tensor& tensor, float stddev, unsigned int& seed) {
    if (stddev <= 0.0f) {
        throw std::invalid_argument("initialize_tensor_normal: stddev must be positive");
    }
    if (seed == 0) {
        throw std::invalid_argument("initialize_tensor_normal: seed must be non-zero");
    }
    // Box–Muller transform: two independent uniform samples u1, u2 give two independent
    // standard normal samples r cos(2 pi u2) and r sin(2 pi u2), where r = sqrt(-2 ln u1).
    for (int i = 0; i < tensor.numel(); i += 2) {
        const float u1 = uniform_01(seed);
        const float u2 = uniform_01(seed);
        const float r = std::sqrt(-2.0f * std::log(u1));
        constexpr float kTwoPi = 6.28318530718f;
        const float theta = kTwoPi * u2;
        const float z0 = r * std::cos(theta);
        const float z1 = r * std::sin(theta);
        tensor.data()[i] = z0 * stddev;
        if (i + 1 < tensor.numel()) {
            tensor.data()[i + 1] = z1 * stddev;
        }
    }
}
