#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

void add_random_infs(std::vector<float> &arr){
    const int arrSize = arr.size();
    if (arrSize == 0) return;
    static std::random_device r;
    static std::default_random_engine e1(r());
    e1.seed(42);
    e1.discard(1000);
    std::uniform_int_distribution<int64_t> uniform_dist(0, arrSize-1);
    
    int64_t numNans = uniform_dist(e1);
    
    for (int64_t i = 0; i < numNans; i++){
        int64_t index = uniform_dist(e1);
        arr.insert(arr.begin() + index, std::numeric_limits<float>::infinity());
    }
}

void add_random_infs(std::vector<double> &arr){
    const int arrSize = arr.size();
    if (arrSize == 0) return;
    static std::random_device r;
    static std::default_random_engine e1(r());
    e1.seed(42);
    e1.discard(1000);
    std::uniform_int_distribution<int64_t> uniform_dist(0, arrSize-1);
    
    int64_t numNans = uniform_dist(e1);
    
    for (int64_t i = 0; i < numNans; i++){
        int64_t index = uniform_dist(e1);
        arr.insert(arr.begin() + index, std::numeric_limits<double>::infinity());
    }
}

void add_random_nans(std::vector<float> &arr){
    const int arrSize = arr.size();
    if (arrSize == 0) return;
    static std::random_device r;
    static std::default_random_engine e1(r());
    e1.seed(42);
    e1.discard(1000);
    std::uniform_int_distribution<int64_t> uniform_dist(0, arrSize-1);
    
    int64_t numNans = uniform_dist(e1);
    
    for (int64_t i = 0; i < numNans; i++){
        int64_t index = uniform_dist(e1);
        arr.insert(arr.begin() + index, std::nanf(""));
    }
}

void add_random_nans(std::vector<double> &arr){
    const int arrSize = arr.size();
    if (arrSize == 0) return;
    static std::random_device r;
    static std::default_random_engine e1(r());
    e1.seed(42);
    e1.discard(1000);
    std::uniform_int_distribution<int64_t> uniform_dist(0, arrSize-1);
    
    int64_t numNans = uniform_dist(e1);
    
    for (int64_t i = 0; i < numNans; i++){
        int64_t index = uniform_dist(e1);
        arr.insert(arr.begin() + index, std::nan(""));
    }
}


template <typename T>
class avx2_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx2_sort);

template <typename T>
class avx2_float_sort : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx2_float_sort);

TYPED_TEST_P(avx2_sort, test_random)
{
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
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_float_sort, test_random_with_specials)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
        
        // Add infinities to both, and nans to only the array given to avx sort
        add_random_infs(arr);
        sortedarr = arr;
        add_random_nans(arr);
        
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        
        // Sort with AVX
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        
        // Validate non NaN part
        for (int i = 0; i < sortedarr.size(); i++){
            if (arr[i] != sortedarr[i]){
                FAIL() << "Array not sorted correctly\n";
            }
        }
        
        // Validate NaNs
        for (int i = sortedarr.size(); i < arr.size(); i++){
            if (!std::isnan(arr[i])){
                FAIL() << "End of array is not all NaNs\n";
            }
        }
        
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_reverse)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* reverse array */
        for (int jj = 0; jj < arrsizes[ii]; ++jj) {
            arr.push_back((TypeParam)(arrsizes[ii] - jj));
        }
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_constant)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* constant array */
        for (int jj = 0; jj < arrsizes[ii]; ++jj) {
            arr.push_back(ii);
        }
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_small_range)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back((TypeParam)(ii + 1));
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        avx2_qsort<TypeParam>(arr.data(), arr.size());
        ASSERT_EQ(sortedarr, arr) << "Array size = " << arrsizes[ii];
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_sort, test_max_value_at_end_of_array)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 1; ii <= 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    for (auto &size : arrsizes) {
        arr = get_uniform_rand_array<TypeParam>(size);
        if (std::numeric_limits<TypeParam>::has_infinity) {
            arr[size - 1] = std::numeric_limits<TypeParam>::infinity();
        }
        else {
            arr[size - 1] = std::numeric_limits<TypeParam>::max();
        }
        sortedarr = arr;
        avx2_qsort(arr.data(), arr.size());
        std::sort(sortedarr.begin(), sortedarr.end());
        EXPECT_EQ(sortedarr, arr) << "Array size = " << size;
        arr.clear();
        sortedarr.clear();
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx2_sort,
                            test_random, test_reverse, test_constant, test_small_range, test_max_value_at_end_of_array);
REGISTER_TYPED_TEST_SUITE_P(avx2_float_sort, test_random_with_specials);

using Types = testing::Types<float, int32_t, uint32_t, double, int64_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx2_sort, Types);

using FloatTypes = testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx2_float_sort, FloatTypes);


template <typename T>
class avx2_select : public ::testing::Test {
};
TYPED_TEST_SUITE_P(avx2_select);

TYPED_TEST_P(avx2_select, test_random)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    std::vector<TypeParam> psortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii]);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        for (size_t k = 0; k < arr.size(); ++k) {
            psortedarr = arr;
            avx2_qselect<TypeParam>(
                    psortedarr.data(), k, psortedarr.size());
            /* index k is correct */
            ASSERT_EQ(sortedarr[k], psortedarr[k]);
            /* Check left partition */
            for (size_t jj = 0; jj < k; jj++) {
                ASSERT_LE(psortedarr[jj], psortedarr[k]);
            }
            /* Check right partition */
            for (size_t jj = k + 1; jj < arr.size(); jj++) {
                ASSERT_GE(psortedarr[jj], psortedarr[k]);
            }
            psortedarr.clear();
        }
        arr.clear();
        sortedarr.clear();
    }
}

TYPED_TEST_P(avx2_select, test_small_range)
{
    std::vector<int64_t> arrsizes;
    for (int64_t ii = 0; ii < 1024; ++ii) {
        arrsizes.push_back(ii);
    }
    std::vector<TypeParam> arr;
    std::vector<TypeParam> sortedarr;
    std::vector<TypeParam> psortedarr;
    for (size_t ii = 0; ii < arrsizes.size(); ++ii) {
        /* Random array */
        arr = get_uniform_rand_array<TypeParam>(arrsizes[ii], 20, 1);
        sortedarr = arr;
        /* Sort with std::sort for comparison */
        std::sort(sortedarr.begin(), sortedarr.end());
        for (size_t k = 0; k < arr.size(); ++k) {
            psortedarr = arr;
            avx2_qselect<TypeParam>(
                    psortedarr.data(), k, psortedarr.size());
            /* index k is correct */
            ASSERT_EQ(sortedarr[k], psortedarr[k]);
            /* Check left partition */
            for (size_t jj = 0; jj < k; jj++) {
                ASSERT_LE(psortedarr[jj], psortedarr[k]);
            }
            /* Check right partition */
            for (size_t jj = k + 1; jj < arr.size(); jj++) {
                ASSERT_GE(psortedarr[jj], psortedarr[k]);
            }
            psortedarr.clear();
        }
        arr.clear();
        sortedarr.clear();
    }
}

REGISTER_TYPED_TEST_SUITE_P(avx2_select, test_random, test_small_range);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx2_select, Types);
