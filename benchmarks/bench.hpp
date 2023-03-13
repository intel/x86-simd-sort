/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "avx512-16bit-qsort.hpp"
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-keyvaluesort.hpp"
#include "avx512-64bit-qsort.hpp"
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

template <typename K, typename V>
struct sorted_t {
    K key;
    V value;
};

static inline uint64_t cycles_start(void)
{
    unsigned a, d;
    __asm__ __volatile__(
            "cpuid\n\t"
            "rdtsc\n\t"
            : "=a"(a), // comma separated output operands
              "=d"(d)
            : // comma separated input operands
            : "rbx", "rcx" // list of clobbered registers
    );
    return (((uint64_t)d << 32) | a);
}

static inline uint64_t cycles_end(void)
{
    unsigned high, low;
    __asm__ __volatile__(
            "rdtscp\n\t"
            "movl %%eax, %[low]\n\t"
            "movl %%edx, %[high]\n\t"
            "cpuid\n\t"
            : [high] "=r"(high), [low] "=r"(low)
            :
            : "rax", "rbx", "rcx", "rdx");
    return (((uint64_t)high << 32) | low);
}

template <typename T>
std::tuple<uint64_t, uint64_t> bench_sort(const std::vector<T> arr,
                                          const uint64_t iters,
                                          const uint64_t lastfew)
{
    std::vector<T> arr_bckup = arr;
    std::vector<uint64_t> runtimes1, runtimes2;
    uint64_t start(0), end(0);
    for (uint64_t ii = 0; ii < iters; ++ii) {
        start = cycles_start();
        avx512_qsort<T>(arr_bckup.data(), arr_bckup.size());
        end = cycles_end();
        runtimes1.emplace_back(end - start);
        arr_bckup = arr;
    }
    uint64_t avx_sort = std::accumulate(runtimes1.end() - lastfew,
                                        runtimes1.end(),
                                        (uint64_t)0)
            / lastfew;

    for (uint64_t ii = 0; ii < iters; ++ii) {
        start = cycles_start();
        std::sort(arr_bckup.begin(), arr_bckup.end());
        end = cycles_end();
        runtimes2.emplace_back(end - start);
        arr_bckup = arr;
    }
    uint64_t std_sort = std::accumulate(runtimes2.end() - lastfew,
                                        runtimes2.end(),
                                        (uint64_t)0)
            / lastfew;
    return std::make_tuple(avx_sort, std_sort);
}

template <typename K, typename V = uint64_t>
std::tuple<uint64_t, uint64_t>
bench_sort_kv(const std::vector<K> keys,
              const std::vector<V> values,
              const std::vector<sorted_t<K, V>> sortedaar,
              const uint64_t iters,
              const uint64_t lastfew)
{

    std::vector<K> keys_bckup = keys;
    std::vector<V> values_bckup = values;
    std::vector<sorted_t<K, V>> sortedaar_bckup = sortedaar;

    std::vector<uint64_t> runtimes1, runtimes2;
    uint64_t start(0), end(0);
    for (uint64_t ii = 0; ii < iters; ++ii) {
        start = cycles_start();
        avx512_qsort_kv<K>(
                keys_bckup.data(), values_bckup.data(), keys_bckup.size());
        end = cycles_end();
        runtimes1.emplace_back(end - start);
        keys_bckup = keys;
        values_bckup = values;
    }
    uint64_t avx_sort = std::accumulate(runtimes1.end() - lastfew,
                                        runtimes1.end(),
                                        (uint64_t)0)
            / lastfew;

    for (uint64_t ii = 0; ii < iters; ++ii) {
        start = cycles_start();
        std::sort(sortedaar_bckup.begin(),
                  sortedaar_bckup.end(),
                  [](sorted_t<K, V> a, sorted_t<K, V> b) {
                      return a.key < b.key;
                  });
        end = cycles_end();
        runtimes2.emplace_back(end - start);
        sortedaar_bckup = sortedaar;
    }
    uint64_t std_sort = std::accumulate(runtimes2.end() - lastfew,
                                        runtimes2.end(),
                                        (uint64_t)0)
            / lastfew;
    return std::make_tuple(avx_sort, std_sort);
}
