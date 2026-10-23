/**
 * @file mnist.cpp
 * @brief MNIST IDX reader.
 *
 * IDX files start with a big-endian header (magic number, item count and, for images, the
 * row and column counts) followed by one unsigned byte per pixel or label.
 */

#include <vadugrad/data/mnist.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kImageMagic = 2051;
constexpr std::int32_t kLabelMagic = 2049;
constexpr int kImageSide = 28;
constexpr int kPixelsPerImage = kImageSide * kImageSide;

std::vector<unsigned char> read_entire_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("MNIST: could not open file: " + path);
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("MNIST: empty file: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file) {
        throw std::runtime_error("MNIST: failed reading file: " + path);
    }
    return bytes;
}

std::int32_t read_u32_big_endian(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return (static_cast<std::int32_t>(bytes[offset]) << 24) |
           (static_cast<std::int32_t>(bytes[offset + 1]) << 16) |
           (static_cast<std::int32_t>(bytes[offset + 2]) << 8) |
           static_cast<std::int32_t>(bytes[offset + 3]);
}

/** Parse an IDX3 image file into [N, 784] pixels scaled to [0, 1]. */
Tensor read_idx_images(const std::string& path) {
    const std::vector<unsigned char> bytes = read_entire_file(path);
    constexpr std::size_t kHeaderSize = 16;
    if (bytes.size() < kHeaderSize) {
        throw std::runtime_error("MNIST: images file too small for header: " + path);
    }
    if (read_u32_big_endian(bytes, 0) != kImageMagic) {
        throw std::runtime_error("MNIST: bad magic number in images file: " + path);
    }
    const int count = read_u32_big_endian(bytes, 4);
    if (read_u32_big_endian(bytes, 8) != kImageSide || read_u32_big_endian(bytes, 12) != kImageSide) {
        throw std::runtime_error("MNIST: expected 28x28 images: " + path);
    }
    if (count <= 0 ||
        bytes.size() < kHeaderSize + static_cast<std::size_t>(count) * kPixelsPerImage) {
        throw std::runtime_error("MNIST: image data truncated: " + path);
    }

    Tensor images({count, kPixelsPerImage});
    for (int i = 0; i < count * kPixelsPerImage; ++i) {
        images.data()[i] = static_cast<float>(bytes[kHeaderSize + static_cast<std::size_t>(i)]) / 255.0f;
    }
    return images;
}

/** Parse an IDX1 label file into [N, 1] class ids. */
Tensor read_idx_labels(const std::string& path) {
    const std::vector<unsigned char> bytes = read_entire_file(path);
    constexpr std::size_t kHeaderSize = 8;
    if (bytes.size() < kHeaderSize) {
        throw std::runtime_error("MNIST: labels file too small for header: " + path);
    }
    if (read_u32_big_endian(bytes, 0) != kLabelMagic) {
        throw std::runtime_error("MNIST: bad magic number in labels file: " + path);
    }
    const int count = read_u32_big_endian(bytes, 4);
    if (count <= 0 || bytes.size() < kHeaderSize + static_cast<std::size_t>(count)) {
        throw std::runtime_error("MNIST: label data truncated: " + path);
    }

    Tensor labels({count, 1});
    for (int i = 0; i < count; ++i) {
        labels.data()[i] = static_cast<float>(bytes[kHeaderSize + static_cast<std::size_t>(i)]);
    }
    return labels;
}

}  // namespace

MnistDataset load_mnist(const std::string& dir, MnistSplit split) {
    const std::string prefix = dir + (split == MnistSplit::Train ? "/train" : "/t10k");
    Tensor images = read_idx_images(prefix + "-images-idx3-ubyte");
    Tensor labels = read_idx_labels(prefix + "-labels-idx1-ubyte");
    if (images.shape()[0] != labels.shape()[0]) {
        throw std::runtime_error("MNIST: image and label counts differ in " + prefix);
    }
    return MnistDataset{std::move(images), std::move(labels)};
}
