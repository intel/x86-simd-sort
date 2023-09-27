/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "test-qsort-common.h"

template <typename T>
class simdsort : public ::testing::Test {
public:
    simdsort()
    {
        std::iota(arrsize.begin(), arrsize.end(), 1);
        arrtype = {"random",
                   "constant",
                   "sorted",
                   "reverse",
                   "smallrange",
                   "max_at_the_end",
                   "rand_max",
                   "rand_with_nan"};
    }
    std::vector<std::string> arrtype;
    std::vector<int64_t> arrsize = std::vector<int64_t>(1024);
};

TYPED_TEST_SUITE_P(simdsort);

TYPED_TEST_P(simdsort, test_qsort)
{
    for (auto type : this->arrtype) {
        for (auto size : this->arrsize) {
            std::vector<TypeParam> arr = get_array<TypeParam>(type, size);
            std::vector<TypeParam> sortedarr = arr;
            std::sort(sortedarr.begin(),
                      sortedarr.end(),
                      compare<TypeParam, std::less<TypeParam>>());
            x86simdsort::qsort(arr.data(), arr.size());
            IS_SORTED(sortedarr, arr, type);
            arr.clear();
            sortedarr.clear();
        }
    }
}

TYPED_TEST_P(simdsort, test_argsort)
{
    for (auto type : this->arrtype) {
        for (auto size : this->arrsize) {
            std::vector<TypeParam> arr = get_array<TypeParam>(type, size);
            std::vector<TypeParam> sortedarr = arr;
            std::sort(sortedarr.begin(),
                      sortedarr.end(),
                      compare<TypeParam, std::less<TypeParam>>());
            auto arg = x86simdsort::argsort(arr.data(), arr.size());
            IS_ARG_SORTED(sortedarr, arr, arg, type);
            arr.clear();
            arg.clear();
        }
    }
}

TYPED_TEST_P(simdsort, test_qselect)
{
    for (auto type : this->arrtype) {
        for (auto size : this->arrsize) {
            int64_t k = rand() % size;
            std::vector<TypeParam> arr = get_array<TypeParam>(type, size);
            std::vector<TypeParam> sortedarr = arr;
            std::nth_element(sortedarr.begin(),
                             sortedarr.begin() + k,
                             sortedarr.end(),
                             compare<TypeParam, std::less<TypeParam>>());
            x86simdsort::qselect(arr.data(), k, arr.size(), true);
            IS_ARR_PARTITIONED(arr, k, sortedarr[k], type);
            arr.clear();
            sortedarr.clear();
        }
    }
}

TYPED_TEST_P(simdsort, test_argselect)
{
    for (auto type : this->arrtype) {
        for (auto size : this->arrsize) {
            int64_t k = rand() % size;
            std::vector<TypeParam> arr = get_array<TypeParam>(type, size);
            std::vector<TypeParam> sortedarr = arr;
            std::sort(sortedarr.begin(),
                      sortedarr.end(),
                      compare<TypeParam, std::less<TypeParam>>());
            auto arg = x86simdsort::argselect(arr.data(), k, arr.size());
            auto arg1 = x86simdsort::argsort(arr.data(), arr.size());
            IS_ARG_PARTITIONED(arr, arg, sortedarr[k], k, type);
            arr.clear();
            sortedarr.clear();
        }
    }
}

TYPED_TEST_P(simdsort, test_partial_qsort)
{
    for (auto type : this->arrtype) {
        for (auto size : this->arrsize) {
            // k should be at least 1
            int64_t k = std::max(0x1l, rand() % size);
            std::vector<TypeParam> arr = get_array<TypeParam>(type, size);
            std::vector<TypeParam> sortedarr = arr;
            std::sort(sortedarr.begin(),
                      sortedarr.end(),
                      compare<TypeParam, std::less<TypeParam>>());
            x86simdsort::partial_qsort(arr.data(), k, arr.size(), true);
            IS_ARR_PARTIALSORTED(arr, k, sortedarr, type);
            arr.clear();
            sortedarr.clear();
        }
    }
}

TYPED_TEST_P(simdsort, test_comparator)
{
#ifdef __FLT16_MAX__
    if constexpr ((std::is_floating_point_v<TypeParam>) || (std::is_same_v<TypeParam, _Float16>)) {
#else
    if constexpr (std::is_floating_point_v<TypeParam>) {
#endif
        auto less = compare<TypeParam, std::less<TypeParam>>();
        auto leq = compare<TypeParam, std::less_equal<TypeParam>>();
        auto greater = compare<TypeParam, std::greater<TypeParam>>();
        auto geq = compare<TypeParam, std::greater_equal<TypeParam>>();
        auto equal = compare<TypeParam, std::equal_to<TypeParam>>();
        TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
        TypeParam inf = std::numeric_limits<TypeParam>::infinity();
        ASSERT_EQ(less(nan, inf), false);
        ASSERT_EQ(less(nan, nan), false);
        ASSERT_EQ(less(inf, nan), true);
        ASSERT_EQ(less(inf, inf), false);
        ASSERT_EQ(leq(nan, inf), false);
        ASSERT_EQ(leq(nan, nan), true);
        ASSERT_EQ(leq(inf, nan), true);
        ASSERT_EQ(leq(inf, inf), true);
        ASSERT_EQ(geq(nan, inf), true);
        ASSERT_EQ(geq(nan, nan), true);
        ASSERT_EQ(geq(inf, nan), false);
        ASSERT_EQ(geq(inf, inf), true);
        ASSERT_EQ(greater(nan, inf), true);
        ASSERT_EQ(greater(nan, nan), false);
        ASSERT_EQ(greater(inf, nan), false);
        ASSERT_EQ(greater(inf, inf), false);
        ASSERT_EQ(equal(nan, inf), false);
        ASSERT_EQ(equal(nan, nan), true);
        ASSERT_EQ(equal(inf, nan), false);
        ASSERT_EQ(equal(inf, inf), true);
    }
}

REGISTER_TYPED_TEST_SUITE_P(simdsort,
                            test_qsort,
                            test_argsort,
                            test_argselect,
                            test_qselect,
                            test_partial_qsort,
                            test_comparator);

using QSortTestTypes = testing::Types<uint16_t,
                                      int16_t,
// support for _Float16 is incomplete in gcc-12
#if __GNUC__ >= 13
                                      _Float16,
#endif
                                      float,
                                      double,
                                      uint32_t,
                                      int32_t,
                                      uint64_t,
                                      int64_t>;

INSTANTIATE_TYPED_TEST_SUITE_P(xss, simdsort, QSortTestTypes);
