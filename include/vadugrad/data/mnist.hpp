#pragma once

/**
 * @file mnist.hpp
 * @brief Reader for the MNIST handwritten-digit files in the binary IDX format.
 */

#include <string>

#include <vadugrad/tensor.hpp>

/** Images and labels of one MNIST split. */
struct MnistDataset {
    Tensor images;  ///< [N, 784]: one flattened 28×28 image per row, pixels scaled to [0, 1].
    Tensor labels;  ///< [N, 1]: digit 0–9 of each image, stored as float.
};

/** Which pair of MNIST files to read. */
enum class MnistSplit {
    Train,  ///< `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` (60,000 images).
    Test,   ///< `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` (10,000 images).
};

/**
 * @brief Load one MNIST split from @p dir.
 * @param dir Directory containing the uncompressed IDX files.
 * @param split Training or test files.
 * @return Images and labels with matching row counts.
 * @throws std::runtime_error if a file is missing, has a wrong magic number or image size,
 *         is truncated, or the image and label counts differ.
 */
[[nodiscard]] MnistDataset load_mnist(const std::string& dir, MnistSplit split);
