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

TEST(avx512_argsort_64bit, test_random)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<int64_t> arr;
        for (auto &size : arrsizes) {
            /* Random array */
            arr = get_uniform_rand_array<int64_t>(size);
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<int64_t>(arr.data(), arr.size());
            std::vector<int64_t> sort1, sort2;
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

TEST(avx512_argsort_64bit, test_constant)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<int64_t> arr;
        for (auto &size : arrsizes) {
            /* constant array */
            auto elem = get_uniform_rand_array<int64_t>(1)[0];
            for (int64_t jj = 0; jj < size; ++jj) {
                arr.push_back(elem);
            }
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<int64_t>(arr.data(), arr.size());
            std::vector<int64_t> sort1, sort2;
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

TEST(avx512_argsort_64bit, test_small_range)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<int64_t> arr;
        for (auto &size : arrsizes) {
            /* array with a smaller range of values */
            arr = get_uniform_rand_array<int64_t>(size, 20, 1);
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<int64_t>(arr.data(), arr.size());
            std::vector<int64_t> sort1, sort2;
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

TEST(avx512_argsort_64bit, test_sorted)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<int64_t> arr;
        for (auto &size : arrsizes) {
            arr = get_uniform_rand_array<int64_t>(size);
            std::sort(arr.begin(), arr.end());
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<int64_t>(arr.data(), arr.size());
            std::vector<int64_t> sort1, sort2;
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

TEST(avx512_argsort_64bit, test_reverse)
{
    if (cpu_has_avx512bw()) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii <= 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<int64_t> arr;
        for (auto &size : arrsizes) {
            arr = get_uniform_rand_array<int64_t>(size);
            std::sort(arr.begin(), arr.end());
            std::reverse(arr.begin(), arr.end());
            std::vector<int64_t> inx1 = std_argsort(arr);
            std::vector<int64_t> inx2
                    = avx512_argsort<int64_t>(arr.data(), arr.size());
            std::vector<int64_t> sort1, sort2;
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
