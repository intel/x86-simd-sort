/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "avx512-16bit-qsort.hpp"
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"
#include "avx512_64bit_keyvaluesort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <gtest/gtest.h>
#include <vector>

template <typename T>
class avx512_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx512_sort);

TYPED_TEST_P(avx512_sort, test_arrsizes)
{
    if (cpu_has_avx512bw()) {
        if ((sizeof(TypeParam) == 2) && (!cpu_has_avx512_vbmi2())) {
            GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
        }
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
<<<<<<< HEAD
            avx512_qsort<TypeParam>(arr.data(), arr.size());
=======
            avx512_qsort<TypeParam>(arr.data(), NULL, arr.size());
>>>>>>> 8873ea16fd047997ae2bf766b85c98125077a74a
            ASSERT_EQ(sortedarr, arr);
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw";
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512_sort, test_arrsizes);

using Types = testing::Types<uint16_t,
                             int16_t,
                             float,
                             double,
                             uint32_t,
                             int32_t,
                             uint64_t,
                             int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(TestPrefix, avx512_sort, Types);

struct sorted_t {
    uint64_t key;
    uint64_t value;
};

bool compare(sorted_t a, sorted_t b)
{
    return a.key == b.key ? a.value < b.value : a.key < b.key;
}
TEST(TestKeyValueSort, KeyValueSort)
{
    std::vector<int64_t> keysizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        keysizes.push_back((uint64_t)ii);
    }
    std::vector<uint64_t> keys;
    std::vector<uint64_t> values;
    std::vector<sorted_t> sortedarr;

    for (size_t ii = 0; ii < keysizes.size(); ++ii) {
        /* Random array */
        keys = get_uniform_rand_array_key(keysizes[ii]);
        //keys = get_uniform_rand_array<uint64_t>(keysizes[ii]);
        values = get_uniform_rand_array<uint64_t>(keysizes[ii]);
        for (size_t i = 0; i < keys.size(); i++) {
            sorted_t tmp_s;
            tmp_s.key = keys[i];
            tmp_s.value = values[i];
            sortedarr.emplace_back(tmp_s);
        }
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end(), compare);
<<<<<<< HEAD
        avx512_qsort_kv<uint64_t>(keys.data(), values.data(), keys.size());
=======
        avx512_qsort<uint64_t>(keys.data(), values.data(), keys.size());
>>>>>>> 8873ea16fd047997ae2bf766b85c98125077a74a
        //ASSERT_EQ(sortedarr, arr);
        for (size_t i = 0; i < keys.size(); i++) {
            ASSERT_EQ(keys[i], sortedarr[i].key);
            ASSERT_EQ(values[i], sortedarr[i].value);
        }
        keys.clear();
        values.clear();
        sortedarr.clear();
    }
}
