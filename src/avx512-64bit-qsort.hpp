/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_QSORT_64BIT
#define AVX512_QSORT_64BIT

#include "avx512-64bit-common.h"
#include "xss-network-qsort.hpp"

template <typename vtype, typename type_t>
static void
qsort_64bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 256) {
        xss::sort_n<vtype, 256>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_64bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 8>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_64bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_64bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
static void qselect_64bit_(type_t *arr,
                           int64_t pos,
                           int64_t left,
                           int64_t right,
                           int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 128) {
        xss::sort_n<vtype, 128>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_64bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 8>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if ((pivot != smallest) && (pos < pivot_index))
        qselect_64bit_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_64bit_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

/* Specialized template function for 64-bit qselect_ funcs*/
template <>
void qselect_<zmm_vector<int64_t>>(
        int64_t *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_64bit_<zmm_vector<int64_t>>(arr, k, left, right, maxiters);
}

template <>
void qselect_<zmm_vector<uint64_t>>(
        uint64_t *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_64bit_<zmm_vector<uint64_t>>(arr, k, left, right, maxiters);
}

template <>
void qselect_<zmm_vector<double>>(
        double *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_64bit_<zmm_vector<double>>(arr, k, left, right, maxiters);
}

/* Specialized template function for 64-bit qsort_ funcs*/
template <>
void qsort_<zmm_vector<int64_t>>(int64_t *arr,
                                 int64_t left,
                                 int64_t right,
                                 int64_t maxiters)
{
    qsort_64bit_<zmm_vector<int64_t>>(arr, left, right, maxiters);
}

template <>
void qsort_<zmm_vector<uint64_t>>(uint64_t *arr,
                                  int64_t left,
                                  int64_t right,
                                  int64_t maxiters)
{
    qsort_64bit_<zmm_vector<uint64_t>>(arr, left, right, maxiters);
}

template <>
void qsort_<zmm_vector<double>>(double *arr,
                                int64_t left,
                                int64_t right,
                                int64_t maxiters)
{
    qsort_64bit_<zmm_vector<double>>(arr, left, right, maxiters);
}
#endif // AVX512_QSORT_64BIT
