#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <vadugrad/data/mnist.hpp>

static void write_file(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(out.good());
}

TEST(MnistLoaderTest, LoadsTinySyntheticDataset) {
    const auto tmp = std::filesystem::temp_directory_path() / "vadugrad_mnist_test";
    std::filesystem::create_directories(tmp);

    // 2 images of 28x28 zeros, labels 3 and 7
    std::vector<unsigned char> img;
    auto be32 = [](int v) {
        return std::vector<unsigned char>{
            static_cast<unsigned char>((v >> 24) & 0xFF), static_cast<unsigned char>((v >> 16) & 0xFF),
            static_cast<unsigned char>((v >> 8) & 0xFF), static_cast<unsigned char>(v & 0xFF)};
    };
    {
        auto h = be32(2051);
        img.insert(img.end(), h.begin(), h.end());
        auto n = be32(2);
        img.insert(img.end(), n.begin(), n.end());
        auto r = be32(28);
        img.insert(img.end(), r.begin(), r.end());
        auto c = be32(28);
        img.insert(img.end(), c.begin(), c.end());
        img.resize(img.size() + 2 * 28 * 28, 0);
    }
    std::vector<unsigned char> lbl;
    {
        auto h = be32(2049);
        lbl.insert(lbl.end(), h.begin(), h.end());
        auto n = be32(2);
        lbl.insert(lbl.end(), n.begin(), n.end());
        lbl.push_back(3);
        lbl.push_back(7);
    }

    write_file((tmp / "train-images-idx3-ubyte").string(), img);
    write_file((tmp / "train-labels-idx1-ubyte").string(), lbl);

    const MnistDataset d = load_mnist_train(tmp.string());
    ASSERT_EQ(d.images.shape(), (std::vector<int>{2, 784}));
    ASSERT_EQ(d.labels.shape(), (std::vector<int>{2, 1}));
    EXPECT_FLOAT_EQ(d.labels({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(d.labels({1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(d.images({0, 0}), 0.0f);
}
