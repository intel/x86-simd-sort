/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/
#ifndef UTILS_RAND_ARRAY
#define UTILS_RAND_ARRAY

#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <algorithm>

template <typename T>
static std::vector<T> get_uniform_rand_array(
        int64_t arrsize,
        T max = std::numeric_limits<T>::max(),
        T min = std::numeric_limits<T>::min())
{
    std::vector<T> arr;
    std::random_device rd;
    if constexpr(std::is_floating_point_v<T>) {
        std::mt19937 gen(rd());
        gen.seed(42);
        std::uniform_real_distribution<T> dis(min, max);
        for (int64_t ii = 0; ii < arrsize; ++ii) {
            arr.emplace_back(dis(gen));
        }
    }
    else if constexpr(std::is_integral_v<T>) {
        std::default_random_engine e1(rd());
        e1.seed(42);
        std::uniform_int_distribution<T> uniform_dist(min, max);
        for (int64_t ii = 0; ii < arrsize; ++ii) {
            arr.emplace_back(uniform_dist(e1));
        }
    }
    return arr;
}

template <typename T>
static std::vector<T>
get_uniform_rand_array_with_uniquevalues(int64_t arrsize,
                                         T max = std::numeric_limits<T>::max(),
                                         T min = std::numeric_limits<T>::min())
{
    std::vector<T> arr = get_uniform_rand_array<T>(arrsize, max, min);
    typename std::vector<T>::iterator ip
            = std::unique(arr.begin(), arr.begin() + arrsize);
    arr.resize(std::distance(arr.begin(), ip));
    return arr;
}

template <typename T>
static std::vector<T>
get_array(std::string arrtype,
          int64_t arrsize,
          T min = std::numeric_limits<T>::min(),
          T max = std::numeric_limits<T>::max())
{
    std::vector<T> arr;
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(arrsize, max, min); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(arrsize, max, min);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1, max, min)[0];
        for (auto ii = 0; ii < arrsize; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(arrsize, max, min);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
    }
    else if (arrtype == "smallrange") {
        arr = get_uniform_rand_array<T>(arrsize, 10, 1);
    }
    else if (arrtype == "max_at_the_end") {
        arr = get_uniform_rand_array<T>(arrsize, max, min);
        if (std::numeric_limits<T>::has_infinity) {
            arr[arrsize - 1] = std::numeric_limits<T>::infinity();
        }
        else {
            arr[arrsize - 1] = std::numeric_limits<T>::max();
        }
    }
    else if (arrtype == "rand_with_nan") {
        arr = get_uniform_rand_array<T>(arrsize, max, min);
        int64_t num_nans = 10 % arrsize;
        std::vector<int64_t> rand_indx
            = get_uniform_rand_array<int64_t>(num_nans, arrsize-1, 0);
        T val;
        if constexpr (std::is_floating_point_v<T>) {
            val = std::numeric_limits<T>::quiet_NaN();
        }
        else {
            val = std::numeric_limits<T>::max();
        }
        for (auto ind : rand_indx) {
            arr[ind] = val;
        }
    }
    else if (arrtype == "rand_max") {
        arr = get_uniform_rand_array<T>(arrsize, max, min);
        T val;
        if constexpr (std::is_floating_point_v<T>) {
            val = std::numeric_limits<T>::infinity();
        }
        else {
            val = std::numeric_limits<T>::max();
        }
        for (auto ii = 1; ii <= arrsize; ++ii) {
            if (rand() % 0x1) {
                arr[ii] = val;
            }
        }
    }
    else {
        std::cout << "Warning: unrecognized array type " << arrtype << std::endl;
    }
    return arr;
}

#endif // UTILS_RAND_ARRAY
