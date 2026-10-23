#include <gtest/gtest.h>
#include <vector>
#include <vadugrad/tensor.hpp>
#include <stdexcept>

TEST(ComputeStridesTest, Basic3D) {
    std::vector<int> shape = {2, 3, 4};
    auto strides = compute_strides(shape);

    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(ComputeStridesTest, SingleDimension) {
    std::vector<int> shape = {5};
    auto strides = compute_strides(shape);

    ASSERT_EQ(strides.size(), 1u);
    EXPECT_EQ(strides[0], 1);
}

TEST(TensorTest, CopyConstructorDoesNotCrash) {
    std::vector<int> shape = {2, 3, 4};
    Tensor a(shape);
    Tensor b = a;  // copy constructor

    SUCCEED();
}

TEST(TensorTest, AssignmentOperatorDoesNotCrash) {
    std::vector<int> shape = {2, 3, 4};
    Tensor a(shape);
    Tensor b(shape);

    b = a;  // assignment operator

    SUCCEED();
}

TEST(TensorOpsTest, Transpose2DWorks) {
    Tensor x({2, 3});
    x({0, 0}) = 1.0f; x({0, 1}) = 2.0f; x({0, 2}) = 3.0f;
    x({1, 0}) = 4.0f; x({1, 1}) = 5.0f; x({1, 2}) = 6.0f;

    Tensor t = transpose2d(x);
    EXPECT_EQ(t.shape(), (std::vector<int>{3, 2}));
    EXPECT_FLOAT_EQ(t({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t({2, 0}), 3.0f);
}

TEST(TensorOpsTest, Transpose2DNonRank2Throws) {
    Tensor x({2, 2, 2});
    EXPECT_THROW({
        Tensor tmp = transpose2d(x);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorOpsTest, MatmulWorks) {
    Tensor a({2, 3});
    Tensor b({3, 2});

    // a = [[1,2,3],[4,5,6]]
    a({0, 0}) = 1.0f; a({0, 1}) = 2.0f; a({0, 2}) = 3.0f;
    a({1, 0}) = 4.0f; a({1, 1}) = 5.0f; a({1, 2}) = 6.0f;

    // b = [[7,8],[9,10],[11,12]]
    b({0, 0}) = 7.0f;  b({0, 1}) = 8.0f;
    b({1, 0}) = 9.0f;  b({1, 1}) = 10.0f;
    b({2, 0}) = 11.0f; b({2, 1}) = 12.0f;

    Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int>{2, 2}));
    EXPECT_FLOAT_EQ(c({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c({1, 1}), 154.0f);
}

TEST(TensorOpsTest, MatmulBadShapeThrows) {
    Tensor a({2, 3});
    Tensor b({4, 2});
    EXPECT_THROW({
        Tensor tmp = matmul(a, b);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorOpsTest, SumAxisKeepdimFalseAndTrue) {
    Tensor x({2, 3});
    x({0, 0}) = 1.0f; x({0, 1}) = 2.0f; x({0, 2}) = 3.0f;
    x({1, 0}) = 4.0f; x({1, 1}) = 5.0f; x({1, 2}) = 6.0f;

    Tensor s0 = sum(x, 0, false);  // [3]
    EXPECT_EQ(s0.shape(), (std::vector<int>{3}));
    EXPECT_FLOAT_EQ(s0({0}), 5.0f);
    EXPECT_FLOAT_EQ(s0({1}), 7.0f);
    EXPECT_FLOAT_EQ(s0({2}), 9.0f);

    Tensor s1k = sum(x, 1, true);  // [2,1]
    EXPECT_EQ(s1k.shape(), (std::vector<int>{2, 1}));
    EXPECT_FLOAT_EQ(s1k({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(s1k({1, 0}), 15.0f);
}

TEST(TensorOpsTest, SumBadAxisThrows) {
    Tensor x({2, 2});
    EXPECT_THROW({
        Tensor tmp = sum(x, -1);
        (void)tmp;
    }, std::invalid_argument);
    EXPECT_THROW({
        Tensor tmp = sum(x, 2);
        (void)tmp;
    }, std::invalid_argument);
}

TEST(TensorTest, FillAndValuesPreservedOnCopy) {
    Tensor a({2, 2});
    a.fill(3.5f);
    Tensor b = a;
    EXPECT_FLOAT_EQ(b({0, 0}), 3.5f);
    EXPECT_FLOAT_EQ(b({1, 1}), 3.5f);
    b({0, 0}) = 1.0f;
    EXPECT_FLOAT_EQ(a({0, 0}), 3.5f);
}

TEST(TensorTest, ReshapePreservesDataOrder) {
    Tensor t({2, 3});
    for (int i = 0; i < 6; ++i) {
        t.data()[i] = static_cast<float>(i + 1);
    }
    t.reshape({3, 2});
    EXPECT_EQ(t.shape(), (std::vector<int>{3, 2}));
    // Row-major: [1,2,3,4,5,6] -> 3x2 [[1,2],[3,4],[5,6]]
    EXPECT_FLOAT_EQ(t({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t({2, 1}), 6.0f);
}

TEST(TensorTest, ReshapeInvalidThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
    EXPECT_THROW(t.reshape({2, 0}), std::invalid_argument);
}

TEST(TensorTest, OffsetOutOfRangeThrows) {
    Tensor t({2, 2});
    EXPECT_THROW({
        (void)t.at(std::vector<int>{2, 0});
    }, std::out_of_range);
    EXPECT_THROW({
        (void)t.at(std::vector<int>{0, -1});
    }, std::out_of_range);
}

TEST(ComputeStridesTest, EmptyShapeThrows) {
    EXPECT_THROW(compute_strides({}), std::invalid_argument);
}

TEST(TensorTest, FixedRankAccessMatchesBracedAccess) {
    Tensor t({2, 3, 4});
    for (int i = 0; i < t.numel(); ++i) {
        t.data()[i] = static_cast<float>(i);
    }
    EXPECT_FLOAT_EQ(t(1, 2, 3), t({1, 2, 3}));
    EXPECT_FLOAT_EQ(t(1, 2, 3), 1 * 12 + 2 * 4 + 3);

    Tensor m({3, 5});
    m(2, 4) = 7.0f;
    EXPECT_FLOAT_EQ(m.data()[2 * 5 + 4], 7.0f);
}

TEST(TensorTest, FixedRankAccessChecksRankAndBounds) {
    Tensor m({3, 5});
    EXPECT_THROW((void)m(3, 0), std::out_of_range);
    EXPECT_THROW((void)m(0, 5), std::out_of_range);
    EXPECT_THROW((void)m(-1, 0), std::out_of_range);
    EXPECT_THROW((void)m(0, 0, 0), std::invalid_argument);
    EXPECT_THROW((void)m({0}), std::invalid_argument);
}

TEST(TensorTest, MoveLeavesSourceEmpty) {
    Tensor a({2, 2});
    a.fill(1.5f);
    Tensor b = std::move(a);
    EXPECT_EQ(b.shape(), (std::vector<int>{2, 2}));
    EXPECT_FLOAT_EQ(b(1, 1), 1.5f);
    EXPECT_EQ(a.ndim(), 0);
    EXPECT_EQ(a.data(), nullptr);
}

// matmul uses a cache-friendly loop order over raw pointers. Compare it with the textbook
// triple loop on odd, non-square sizes, where an indexing mistake cannot hide behind symmetry.
TEST(TensorOpsTest, MatmulMatchesNaiveTripleLoop) {
    constexpr int m = 37, k = 53, n = 29;
    Tensor a({m, k});
    Tensor b({k, n});
    for (int i = 0; i < a.numel(); ++i) a.data()[i] = static_cast<float>((i * 7) % 11) - 5.0f;
    for (int i = 0; i < b.numel(); ++i) b.data()[i] = static_cast<float>((i * 5) % 13) - 6.0f;

    const Tensor c = matmul(a, b);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (int p = 0; p < k; ++p) {
                expected += a(i, p) * b(p, j);
            }
            ASSERT_FLOAT_EQ(c(i, j), expected) << "i=" << i << " j=" << j;
        }
    }
}
