/*******************************************
 * * Copyright (C) 2023 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "avx512-64bit-argsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

template <typename T>
class avx512argsort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512argsort);

template <typename T>
std::vector<int64_t> std_argsort(const std::vector<T> &array)
{
    std::vector<int64_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array sizeent
                  return array[left] < array[right];
              });

    return indices;
}

TYPED_TEST_P(avx512argsort, test_random)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        for (auto &size : arrsizes) {
            /* Random array */
            arr = get_uniform_rand_array<TypeParam>(size);
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<TypeParam>(arr.data(), arr.size());
            std::vector<TypeParam> sort1, sort2;
            for (size_t jj = 0; jj < size; ++jj) {
                sort1.push_back(arr[inx1[jj]]);
                sort2.push_back(arr[inx2[jj]]);
            }
            ASSERT_EQ(sort1, sort2) << "Array size =" << size;
            arr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

TYPED_TEST_P(avx512argsort, test_constant)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        for (auto &size : arrsizes) {
            /* constant array */
            auto elem = get_uniform_rand_array<TypeParam>(1)[0];
            for (int64_t jj = 0; jj < size; ++jj) {
                arr.push_back(elem);
            }
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<TypeParam>(arr.data(), arr.size());
            std::vector<TypeParam> sort1, sort2;
            for (size_t jj = 0; jj < size; ++jj) {
                sort1.push_back(arr[inx1[jj]]);
                sort2.push_back(arr[inx2[jj]]);
            }
            ASSERT_EQ(sort1, sort2) << "Array size =" << size;
            arr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

TYPED_TEST_P(avx512argsort, test_small_range)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        for (auto &size : arrsizes) {
            /* array with a smaller range of values */
            arr = get_uniform_rand_array<TypeParam>(size, 20, 1);
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<TypeParam>(arr.data(), arr.size());
            std::vector<TypeParam> sort1, sort2;
            for (size_t jj = 0; jj < size; ++jj) {
                sort1.push_back(arr[inx1[jj]]);
                sort2.push_back(arr[inx2[jj]]);
            }
            ASSERT_EQ(sort1, sort2) << "Array size = " << size;
            arr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

TYPED_TEST_P(avx512argsort, test_sorted)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        for (auto &size : arrsizes) {
            arr = get_uniform_rand_array<TypeParam>(size);
            std::sort(arr.begin(), arr.end());
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<TypeParam>(arr.data(), arr.size());
            std::vector<TypeParam> sort1, sort2;
            for (size_t jj = 0; jj < size; ++jj) {
                sort1.push_back(arr[inx1[jj]]);
                sort2.push_back(arr[inx2[jj]]);
            }
            ASSERT_EQ(sort1, sort2) << "Array size =" << size;
            arr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

TYPED_TEST_P(avx512argsort, test_reverse)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<TypeParam> arr;
        for (auto &size : arrsizes) {
            arr = get_uniform_rand_array<TypeParam>(size);
            std::sort(arr.begin(), arr.end());
            std::reverse(arr.begin(), arr.end());
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<TypeParam>(arr.data(), arr.size());
            std::vector<TypeParam> sort1, sort2;
            for (size_t jj = 0; jj < size; ++jj) {
                sort1.push_back(arr[inx1[jj]]);
                sort2.push_back(arr[inx2[jj]]);
            }
            ASSERT_EQ(sort1, sort2) << "Array size =" << size;
            arr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512argsort,
                            test_random,
                            test_reverse,
                            test_constant,
                            test_sorted,
                            test_small_range);

using ArgSortTestTypes = testing::Types<int32_t,
                                        uint32_t,
                                        float,
                                        uint64_t,
                                        int64_t,
                                        double>;

INSTANTIATE_TYPED_TEST_SUITE_P(TestPrefix, avx512argsort, ArgSortTestTypes);
