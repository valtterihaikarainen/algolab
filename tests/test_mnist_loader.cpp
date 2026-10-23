#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vadugrad/data/mnist.hpp>

namespace {

void append_u32_big_endian(std::vector<unsigned char>& bytes, int value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xFF));
    }
}

void write_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(out.good());
}

/** Write a two-image IDX pair with the given file prefix ("train" or "t10k"). */
std::filesystem::path write_tiny_split(const std::string& prefix, int image_magic) {
    const auto dir = std::filesystem::temp_directory_path() / "vadugrad_mnist_test";
    std::filesystem::create_directories(dir);

    std::vector<unsigned char> images;
    append_u32_big_endian(images, image_magic);
    append_u32_big_endian(images, 2);
    append_u32_big_endian(images, 28);
    append_u32_big_endian(images, 28);
    images.resize(images.size() + 2 * 784, 0);
    images[16 + 784 + 5] = 255;  // second image, pixel 5 fully white

    std::vector<unsigned char> labels;
    append_u32_big_endian(labels, 2049);
    append_u32_big_endian(labels, 2);
    labels.push_back(3);
    labels.push_back(7);

    write_file(dir / (prefix + "-images-idx3-ubyte"), images);
    write_file(dir / (prefix + "-labels-idx1-ubyte"), labels);
    return dir;
}

}  // namespace

TEST(MnistLoaderTest, LoadsTrainSplitAndScalesPixels) {
    const auto dir = write_tiny_split("train", 2051);
    const MnistDataset data = load_mnist(dir.string(), MnistSplit::Train);
    ASSERT_EQ(data.images.shape(), (std::vector<int>{2, 784}));
    ASSERT_EQ(data.labels.shape(), (std::vector<int>{2, 1}));
    EXPECT_FLOAT_EQ(data.labels(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(data.labels(1, 0), 7.0f);
    EXPECT_FLOAT_EQ(data.images(1, 5), 1.0f);
    EXPECT_FLOAT_EQ(data.images(0, 5), 0.0f);
}

TEST(MnistLoaderTest, LoadsTestSplitFromT10kFiles) {
    const auto dir = write_tiny_split("t10k", 2051);
    const MnistDataset data = load_mnist(dir.string(), MnistSplit::Test);
    EXPECT_EQ(data.images.shape()[0], 2);
}

TEST(MnistLoaderTest, RejectsBadMagicAndMissingFiles) {
    const auto dir = write_tiny_split("train", 1234);
    EXPECT_THROW((void)load_mnist(dir.string(), MnistSplit::Train), std::runtime_error);
    EXPECT_THROW((void)load_mnist("/nonexistent-dir", MnistSplit::Train), std::runtime_error);
}
