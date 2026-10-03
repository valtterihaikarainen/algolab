#pragma once

/**
 * @file mnist.hpp
 * @brief Minimal MNIST IDX file reader (images + labels) into row-major @ref Tensor batches.
 */

#include <string>

#include <vadugrad/tensor.hpp>

struct MnistDataset {
    Tensor images;  ///< [N, 784] float in [0,1]
    Tensor labels;  ///< [N, 1] float storing integer class id
};

/**
 * @brief Load MNIST training images/labels from classic IDX files.
 *
 * Expected filenames in @p dir:
 * - `train-images-idx3-ubyte`
 * - `train-labels-idx1-ubyte`
 */
[[nodiscard]] MnistDataset load_mnist_train(const std::string& dir);
