/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX2_QSORT_32BIT
#define AVX2_QSORT_32BIT

#include "avx2-32bit-common.h"
#include "avx2-network-qsort.hpp"

// Assumes ymm is bitonic and performs a recursive half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t bitonic_merge_ymm_32bit(ymm_t ymm)
{
    
    const typename vtype::opmask_t oxAA = _mm256_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
    const typename vtype::opmask_t oxCC = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
    const typename vtype::opmask_t oxF0 = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0);

    // 1) half_cleaner[8]: compare 0-4, 1-5, 2-6, 3-7
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(_mm256_set_epi32(NETWORK_32BIT_4), ymm),
            oxF0);
    // 2) half_cleaner[4]
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(_mm256_set_epi32(NETWORK_32BIT_3), ymm),
            oxCC);
    // 3) half_cleaner[1]
    ymm = cmp_merge<vtype>(
            ymm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
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
    if (right + 1 - left <= 512) {
        sort_n<vtype, 512>(arr + left, (int32_t)(right + 1 - left));
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
    if (right + 1 - left <= 128) {
        sort_n<vtype, 128>(arr + left, (int32_t)(right + 1 - left));
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
void avx2_qselect<int32_t>(int32_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_32bit_<ymm_vector<int32_t>, int32_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qselect<uint32_t>(uint32_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_32bit_<ymm_vector<uint32_t>, uint32_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qselect<float>(float *arr, int64_t k, int64_t arrsize, bool hasnan)
{
    int64_t indx_last_elem = arrsize - 1;
    if (UNLIKELY(hasnan)) {
         indx_last_elem = move_nans_to_end_of_array(arr, arrsize);
    }
    if (indx_last_elem >= k) {
        qselect_32bit_<ymm_vector<float>, float>(
            arr, k, 0, indx_last_elem, 2 * (int64_t)log2(indx_last_elem));
    }
}

template <>
void avx2_qsort<int32_t>(int32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<ymm_vector<int32_t>, int32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<uint32_t>(uint32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<ymm_vector<uint32_t>, uint32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<float>(float *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_32bit_<ymm_vector<float>, float>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // AVX2_QSORT_32BIT
