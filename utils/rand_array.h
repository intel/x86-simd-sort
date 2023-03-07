/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include <random>
#include <type_traits>
#include <vector>

template <typename T>
static std::vector<T> get_uniform_rand_array(
        int64_t arrsize,
        int seed = 42,
        T max = std::numeric_limits<T>::max(),
        T min = std::numeric_limits<T>::min(),
        typename std::enable_if<std::is_integral<T>::value>::type * = 0)
{
    std::vector<T> arr;
    std::random_device r;
    std::default_random_engine e1(r());
    e1.seed(seed);
    std::uniform_int_distribution<T> uniform_dist(min, max);
    for (int64_t ii = 0; ii < arrsize; ++ii) {
        arr.emplace_back(uniform_dist(e1));
    }
    return arr;
}

template <typename T>
static std::vector<T> get_uniform_rand_array(
        int64_t arrsize,
        int seed = 42,
        T max = std::numeric_limits<T>::max(),
        T min = std::numeric_limits<T>::min(),
        typename std::enable_if<std::is_floating_point<T>::value>::type * = 0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(seed);
    std::uniform_real_distribution<> dis(min, max);
    std::vector<T> arr;
    for (int64_t ii = 0; ii < arrsize; ++ii) {
        arr.emplace_back(dis(gen));
    }
    return arr;
}
