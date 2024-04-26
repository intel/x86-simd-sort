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
                   "random_5d",
                   "rand_max"};
    }
    std::vector<std::string> arrtype;
    std::vector<size_t> arrsize = std::vector<size_t>(1024);
};

TYPED_TEST_SUITE_P(simdkvsort);

template <typename T>
bool same_values(T* v1, T* v2, size_t size){
    // Checks that the values are the same except (maybe) their ordering
    auto cmp_eq = compare<T, std::equal_to<T>>();
    
    // TODO hardcoding hasnan to true doesn't break anything right?
    x86simdsort::qsort(v1, size, true);
    x86simdsort::qsort(v2, size, true);
    
    for (size_t i = 0; i < size; i++){
        if (!cmp_eq(v1[i], v2[i])){
            return false;
        }
    }
    
    return true;
}

template <typename T1, typename T2>
bool kv_equivalent(T1* keys_comp, T2* vals_comp, T1* keys_ref, T2* vals_ref, size_t size){
    auto cmp_eq = compare<T1, std::equal_to<T1>>();
    
    // First check keys are exactly identical
    for (size_t i = 0; i < size; i++){
        if (!cmp_eq(keys_comp[i], keys_ref[i])){
            return false;
        }
    }
    
    size_t i_start = 0;
    T1 key_start = keys_comp[0];
    // Loop through all identical keys in a block, then compare the sets of values to make sure they are identical
    // We need the index after the loop
    size_t i = 0;
    for (; i < size; i++){
        if (!cmp_eq(keys_comp[i], key_start)){
            // Check that every value in 

            if (!same_values(vals_ref + i_start, vals_comp + i_start, i - i_start)){
                return false;
            }
            
            // Now setup the start variables to begin gathering keys for the next group
            i_start = i;
            key_start = keys_comp[i];
        }
    }
    
    return true;
}

TYPED_TEST_P(simdkvsort, test_kvsort)
{
    using T1 = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using T2 = typename std::tuple_element<1, decltype(TypeParam())>::type;
    for (auto type : this->arrtype) {
        bool hasnan = (type == "rand_with_nan") ? true : false;
        for (auto size : this->arrsize) {
            std::vector<T1> key = get_array<T1>(type, size);
            std::vector<T2> val = get_array<T2>(type, size);
            std::vector<T1> key_bckp = key;
            std::vector<T2> val_bckp = val;
            x86simdsort::keyvalue_qsort(key.data(), val.data(), size, hasnan);
            xss::scalar::keyvalue_qsort(
                    key_bckp.data(), val_bckp.data(), size, hasnan);
            
            bool is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), size);
            ASSERT_EQ(is_kv_equivalent, true);
            
            key.clear();
            val.clear();
            key_bckp.clear();
            val_bckp.clear();
        }
    }
}

TYPED_TEST_P(simdkvsort, test_validator)
{
    // Tests a few edge cases to verify the tests are working correctly and identifying it as functional
    using T1 = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using T2 = typename std::tuple_element<1, decltype(TypeParam())>::type;
    
    bool is_kv_equivalent;
    
    std::vector<T1> key = {0, 0, 1, 1};
    std::vector<T2> val = {1, 2, 3, 4};
    std::vector<T1> key_bckp = key;
    std::vector<T2> val_bckp = val;
    
    // Duplicate keys, but otherwise exactly identical
    is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), key.size());
    ASSERT_EQ(is_kv_equivalent, true);
    
    val = {2,1,4,3};
    
    // Now values are backwards, but this is still fine
    is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), key.size());
    ASSERT_EQ(is_kv_equivalent, true);
    
    val = {1,3,2,4};
    
    // Now values are mixed up, should fail
    is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), key.size());
    ASSERT_EQ(is_kv_equivalent, false);
    
    val = {1,2,3,4};
    key = {0,0,0,0};
    
    // Now keys are messed up, should fail
    is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), key.size());
    ASSERT_EQ(is_kv_equivalent, false);
    
    key = {0,0,0,0,0,0};
    key_bckp = key;
    val_bckp = {1,2,3,4,5,6};
    val = {4,3,1,6,5,2};
    
    // All keys identical, simply reordered values
    is_kv_equivalent = kv_equivalent<T1, T2>(key.data(), val.data(), key_bckp.data(), val_bckp.data(), key.size());
    ASSERT_EQ(is_kv_equivalent, true);
}

REGISTER_TYPED_TEST_SUITE_P(simdkvsort, test_kvsort, test_validator);

#define CREATE_TUPLES(type) \
    std::tuple<double, type>, std::tuple<uint64_t, type>, \
            std::tuple<int64_t, type>, std::tuple<float, type>, \
            std::tuple<uint32_t, type>, std::tuple<int32_t, type>

using QKVSortTestTypes = testing::Types<CREATE_TUPLES(double),
                                        CREATE_TUPLES(uint64_t),
                                        CREATE_TUPLES(int64_t),
                                        CREATE_TUPLES(uint32_t),
                                        CREATE_TUPLES(int32_t),
                                        CREATE_TUPLES(float)>;

INSTANTIATE_TYPED_TEST_SUITE_P(xss, simdkvsort, QKVSortTestTypes);
