/*******************************************
 * * Copyright (C) 2023 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

template <typename T>
class avx512argselect : public ::testing::Test {
};

TYPED_TEST_SUITE_P(avx512argselect);

TYPED_TEST_P(avx512argselect, test_random)
{
    if (__builtin_cpu_supports("avx512bw")) {
        const int arrsize = 1024;
        auto arr = get_uniform_rand_array<TypeParam>(arrsize);
        std::vector<int64_t> sorted_inx;
        if (std::is_floating_point<TypeParam>::value) {
            arr[0] = std::numeric_limits<TypeParam>::quiet_NaN();
            arr[1] = std::numeric_limits<TypeParam>::quiet_NaN();
        }
        sorted_inx = std_argsort(arr);
        std::vector<int64_t> kth;
        for (int64_t ii = 0; ii < arrsize - 3; ++ii) {
            kth.push_back(ii);
        }
        for (auto &k : kth) {
            std::vector<int64_t> inx
                    = avx512_argselect<TypeParam>(arr.data(), k, arr.size());
            auto true_kth = arr[sorted_inx[k]];
            EXPECT_EQ(true_kth, arr[inx[k]]) << "Failed at index k = " << k;
            if (k >= 1) {
                EXPECT_GE(true_kth, std_max_element(arr, inx, 0, k - 1))
                        << "failed at k = " << k;
            }
            if (k != arrsize - 1) {
                EXPECT_LE(true_kth,
                          std_min_element(arr, inx, k + 1, arrsize - 1))
                        << "failed at k = " << k;
            }
            EXPECT_UNIQUE(inx)
        }
    }
    else {
        GTEST_SKIP() << "Skipping this test, it requires avx512bw ISA";
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx512argselect, test_random);
