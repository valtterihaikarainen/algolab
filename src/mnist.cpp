/**
 * @file mnist.cpp
 * @brief Minimal MNIST IDX reader.
 */

#include <vadugrad/data/mnist.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<unsigned char> read_entire_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("MNIST: could not open file: " + path);
    }
    f.seekg(0, std::ios::end);
    const std::streamoff sz = f.tellg();
    if (sz <= 0) {
        throw std::runtime_error("MNIST: empty file: " + path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f) {
        throw std::runtime_error("MNIST: failed reading file: " + path);
    }
    return buf;
}

}  // namespace

MnistDataset load_mnist_train(const std::string& dir) {
    const std::string img_path = dir + "/train-images-idx3-ubyte";
    const std::string lbl_path = dir + "/train-labels-idx1-ubyte";

    const std::vector<unsigned char> img_bytes = read_entire_file(img_path);
    const std::vector<unsigned char> lbl_bytes = read_entire_file(lbl_path);

    std::vector<unsigned char> img = img_bytes;
    std::vector<unsigned char> lbl = lbl_bytes;

    auto u32_be = [](const std::vector<unsigned char>& bytes, int off) {
        return (static_cast<int32_t>(bytes[static_cast<std::size_t>(off)]) << 24) |
               (static_cast<int32_t>(bytes[static_cast<std::size_t>(off + 1)]) << 16) |
               (static_cast<int32_t>(bytes[static_cast<std::size_t>(off + 2)]) << 8) |
               static_cast<int32_t>(bytes[static_cast<std::size_t>(off + 3)]);
    };

    auto read_image_header = [&](std::vector<unsigned char>& bytes) {
        if (bytes.size() < 16) {
            throw std::runtime_error("MNIST: images file too small for header");
        }
        const int32_t magic = u32_be(bytes, 0);
        if (magic != 2051) {
            throw std::runtime_error("MNIST: bad magic in images");
        }
        const int32_t n = u32_be(bytes, 4);
        const int32_t rows = u32_be(bytes, 8);
        const int32_t cols = u32_be(bytes, 12);
        bytes.erase(bytes.begin(), bytes.begin() + 16);
        return std::tuple<int32_t, int32_t, int32_t>{n, rows, cols};
    };

    auto read_label_header = [&](std::vector<unsigned char>& bytes) {
        // IDX1: 4-byte magic + 4-byte number of items (no extra dims).
        if (bytes.size() < 8) {
            throw std::runtime_error("MNIST: labels file too small for header");
        }
        const int32_t magic = u32_be(bytes, 0);
        if (magic != 2049) {
            throw std::runtime_error("MNIST: bad magic in labels");
        }
        const int32_t n = u32_be(bytes, 4);
        bytes.erase(bytes.begin(), bytes.begin() + 8);
        return n;
    };

    const auto img_hdr = read_image_header(img);
    const int32_t num_images = std::get<0>(img_hdr);
    const int32_t rows = std::get<1>(img_hdr);
    const int32_t cols = std::get<2>(img_hdr);
    if (rows != 28 || cols != 28) {
        throw std::runtime_error("MNIST: expected 28x28 images");
    }
    const std::size_t expected_img_payload = static_cast<std::size_t>(num_images) * 28u * 28u;
    if (img.size() < expected_img_payload) {
        throw std::runtime_error("MNIST: image payload truncated");
    }
    img.resize(expected_img_payload);

    const int32_t num_labels = read_label_header(lbl);
    if (num_labels != num_images) {
        throw std::runtime_error("MNIST: label/image count mismatch");
    }
    if (lbl.size() < static_cast<std::size_t>(num_labels)) {
        throw std::runtime_error("MNIST: label payload truncated");
    }
    lbl.resize(static_cast<std::size_t>(num_labels));

    Tensor images({num_images, 784});
    Tensor labels({num_images, 1});

    for (int i = 0; i < num_images; ++i) {
        for (int p = 0; p < 784; ++p) {
            const unsigned char px =
                img[static_cast<std::size_t>(i) * 784u + static_cast<std::size_t>(p)];
            images({i, p}) = static_cast<float>(px) / 255.0f;
        }
        labels({i, 0}) = static_cast<float>(lbl[static_cast<std::size_t>(i)]);
    }

    return MnistDataset{images, labels};
}
