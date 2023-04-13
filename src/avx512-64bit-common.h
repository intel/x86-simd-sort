/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_64BIT_COMMON
#define AVX512_64BIT_COMMON
#include "avx512-common-qsort.h"

/*
 * Constants used in sorting 8 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM                  7, 6, 5, 4, 3, 2, 1, 0
#define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4

X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(double *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask8 loadmask = 0xFF;
    while (arrsize > 0) {
        if (arrsize < 8) { loadmask = (0x01 << arrsize) - 0x01; }
        __m512d in_zmm = _mm512_maskz_loadu_pd(loadmask, arr);
        __mmask8 nanmask = _mm512_cmp_pd_mask(in_zmm, in_zmm, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm512_mask_storeu_pd(arr, nanmask, ZMM_MAX_DOUBLE);
        arr += 8;
        arrsize -= 8;
    }
    return nan_count;
}

X86_SIMD_SORT_INLINE void
replace_inf_with_nan(double *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = std::nan("1");
        nan_count -= 1;
    }
}
/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t sort_zmm_64bit(zmm_t zmm)
{
    const __m512i rev_index = _mm512_set_epi64(NETWORK_64BIT_2);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_1), zmm),
            0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(zmm, vtype::permutexvar(rev_index, zmm), 0xF0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_3), zmm),
            0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    return zmm;
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_64bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 8
    int64_t size = (right - left) / 8;
    using zmm_t = typename vtype::zmm_t;
    __m512i rand_index = _mm512_set_epi64(left + size,
                                          left + 2 * size,
                                          left + 3 * size,
                                          left + 4 * size,
                                          left + 5 * size,
                                          left + 6 * size,
                                          left + 7 * size,
                                          left + 8 * size);
    zmm_t rand_vec = vtype::template i64gather<sizeof(type_t)>(rand_index, arr);
    // pivot will never be a nan, since there are no nan's!
    zmm_t sort = sort_zmm_64bit<vtype>(rand_vec);
    return ((type_t *)&sort)[4];
}

#endif
