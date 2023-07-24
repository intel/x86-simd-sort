#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <gtest/gtest.h>

template <typename T>
class avx2_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx2_sort);

TYPED_TEST_P(avx2_sort, test_random)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_reverse)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* reverse array */
        for (int jj = 0; jj < arrsizes[ii]; ++jj) {
            arr.push_back((TypeParam)(arrsizes[ii] - jj));
        }
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_constant)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* constant array */
        for (int jj = 0; jj < arrsizes[ii]; ++jj) {
            arr.push_back(ii);
        }
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_small_range)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_max_value_at_end_of_array)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 1; ii <= 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (auto &size : arrsizes) {
        arr = get_uniform_rand_array<TypeParam>(size);
        if (std::numeric_limits<TypeParam>::has_infinity) {
            arr[size - 1] = std::numeric_limits<TypeParam>::infinity();
        }
        else {
            arr[size - 1] = std::numeric_limits<TypeParam>::max();
        }
        sortedarr = arr;
        avx2_qsort(arr.data(), arr.size());
        std::sort(sortedarr.begin(), sortedarr.end());
        EXPECT_EQ(sortedarr, arr) << "Array size = " << size;
        arr.clear();
        sortedarr.clear();
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx2_sort,
                            test_random, test_reverse, test_constant, test_small_range, test_max_value_at_end_of_array);

using Types = testing::Types<float, int32_t, uint32_t, double, int64_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx2_sort, Types);

template <typename T>
class avx2_select : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx2_select);

TYPED_TEST_P(avx2_select, test_random)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    std::vector<TypeParam> psortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        for (size_t k = 0; k < arr.size(); ++k) {
            psortedarr = arr;
            avx2_qselect<TypeParam>(
                    psortedarr.data(), k, psortedarr.size());
            /* index k is correct */
            ASSERT_EQ(sortedarr[k], psortedarr[k]);
            /* Check left partition */
            for (size_t jj = 0; jj < k; jj++) {
                ASSERT_LE(psortedarr[jj], psortedarr[k]);
            }
            /* Check right partition */
            for (size_t jj = k + 1; jj < arr.size(); jj++) {
                ASSERT_GE(psortedarr[jj], psortedarr[k]);
            }
            psortedarr.clear();
        }
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_select, test_small_range)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    std::vector<TypeParam> psortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        for (size_t k = 0; k < arr.size(); ++k) {
            psortedarr = arr;
            avx2_qselect<TypeParam>(
                    psortedarr.data(), k, psortedarr.size());
            /* index k is correct */
            ASSERT_EQ(sortedarr[k], psortedarr[k]);
            /* Check left partition */
            for (size_t jj = 0; jj < k; jj++) {
                ASSERT_LE(psortedarr[jj], psortedarr[k]);
            }
            /* Check right partition */
            for (size_t jj = k + 1; jj < arr.size(); jj++) {
                ASSERT_GE(psortedarr[jj], psortedarr[k]);
            }
            psortedarr.clear();
        }
        arr.clear();
        sortedarr.clear();
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx2_select, test_random, test_small_range);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx2_select, Types);
