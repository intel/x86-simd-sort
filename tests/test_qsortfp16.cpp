/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include <gtest/gtest.h>
#include <vector>
#include "avx512fp16-16bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"

TEST(avx512_qsort_float16, test_arrsizes)
{
    if ((cpu_has_avx512bw()) && (cpu_has_avx512_vbmi2())) {
        std::vector<int64_t> arrsizes;
        for (int64_t ii = 0; ii < 1024; ++ii) {
            arrsizes.push_back(ii);
        }
        std::vector<_Float16> arr;
        std::vector<_Float16> sortedarr;
        for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
            /* Random array */
            std::vector<uint16_t> temp =
                get_uniform_rand_array<uint16_t>(arrsizes[ii]);
            arr.reserve(arrsizes[ii]);
            memcpy(arr.data(), temp.data(), arrsizes[ii]*2);
            sortedarr = arr;
            /* Sort with std::sort for comparison */
            std::sort(sortedarr.begin(), sortedarr.end());
            avx512_qsort<_Float16>(arr.data(), arr.size());
            ASSERT_EQ(sortedarr, arr);
            arr.clear();
            sortedarr.clear();
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512_vbmi2";
    }
}
