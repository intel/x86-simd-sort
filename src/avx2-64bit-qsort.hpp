/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX2_QSORT_64BIT
#define AVX2_QSORT_64BIT

#include "avx2-64bit-common.h"
#include "avx2-network-qsort.hpp"

#define PERMUTE_MASK_IMPL(a,b,c,d) (SHUFFLE_MASK(a,b,c,d))
#define PERMUTE_MASK(...) PERMUTE_MASK_IMPL(__VA_ARGS__)

// Assumes ymm is bitonic and performs a recursive half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t bitonic_merge_ymm_32bit(ymm_t ymm)
{
    const typename vtype::opmask_t oxAA = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0);
    const typename vtype::opmask_t oxCC = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0);
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::template permutexvar<SHUFFLE_MASK(1,0,3,2)>(ymm), oxCC);
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::template permutexvar<SHUFFLE_MASK(2,3,0,1)>(ymm), oxAA);
    return ymm;
}

template <typename vtype, typename type_t>
static void
qsort_32bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
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
        sort_n<vtype, 128>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_32bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 8>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_32bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_32bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
static void qselect_32bit_(type_t *arr,
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
    if (right + 1 - left <= 64) {
        sort_n<vtype, 64>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_32bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 8>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if ((pivot != smallest) && (pos < pivot_index))
        qselect_32bit_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_32bit_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

template <>
void avx2_qselect<int64_t>(int64_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_32bit_<ymm_vector<int64_t>, int64_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qselect<uint64_t>(uint64_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_32bit_<ymm_vector<uint64_t>, uint64_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qselect<double>(double *arr, int64_t k, int64_t arrsize, bool hasnan)
{
    int64_t indx_last_elem = arrsize - 1;
    if (UNLIKELY(hasnan)) {
         indx_last_elem = move_nans_to_end_of_array(arr, arrsize);
    }
    if (indx_last_elem >= k) {
        qselect_32bit_<ymm_vector<double>, double>(
            arr, k, 0, indx_last_elem, 2 * (int64_t)log2(indx_last_elem));
    }
}

template <>
void avx2_qsort<int64_t>(int64_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<ymm_vector<int64_t>, int64_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<uint64_t>(uint64_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<ymm_vector<uint64_t>, uint64_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<double>(double *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_32bit_<ymm_vector<double>, double>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // AVX2_QSORT_32BIT
