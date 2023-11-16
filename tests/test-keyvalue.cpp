/*******************************************
 * * Copyright (C) 2022-2023 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "rand_array.h"
#include "x86simdsort.h"
#include "x86simdsort-scalar.h"
#include <gtest/gtest.h>

template <typename T>
class simdkvsort : public ::testing::Test {
public:
    simdkvsort()
    {
        std::iota(arrsize.begin(), arrsize.end(), 1);
        arrtype = {"random",
                   "constant",
                   "sorted",
                   "reverse",
                   "smallrange",
                   "max_at_the_end",
                   "rand_max"};
    }
    std::vector<std::string> arrtype;
    std::vector<size_t> arrsize = std::vector<size_t>(1024);
};

TYPED_TEST_SUITE_P(simdkvsort);

TYPED_TEST_P(simdkvsort, test_kvsort)
{
    using T1 = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using T2 = typename std::tuple_element<1, decltype(TypeParam())>::type;
    for (auto type : this->arrtype) {
        bool hasnan = (type == "rand_with_nan") ? true : false;
        for (auto size : this->arrsize) {
            std::vector<T1> v1 = get_array<T1>(type, size);
            std::vector<T2> v2 = get_array<T2>(type, size);
            std::vector<T1> v1_bckp = v1;
            std::vector<T2> v2_bckp = v2;
            xss::scalar::keyvalue_qsort(v1_bckp.data(), v2_bckp.data(), size, hasnan);
            x86simdsort::keyvalue_qsort(v1.data(), v2.data(), size, hasnan);
            ASSERT_EQ(v1, v1_bckp);
            ASSERT_EQ(v2, v2_bckp);
            v1.clear(); v2.clear();
            v1_bckp.clear(); v2_bckp.clear();
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(simdkvsort, test_kvsort);

using QKVSortTestTypes = testing::Types<std::tuple<double, double>,
                                      std::tuple<double, uint64_t>,
                                      std::tuple<double, int64_t>,
                                      std::tuple<uint64_t, double>,
                                      std::tuple<uint64_t, uint64_t>,
                                      std::tuple<uint64_t, int64_t>,
                                      std::tuple<int64_t, double>,
                                      std::tuple<int64_t, uint64_t>,
                                      std::tuple<int64_t, int64_t>>;

INSTANTIATE_TYPED_TEST_SUITE_P(xss, simdkvsort, QKVSortTestTypes);
