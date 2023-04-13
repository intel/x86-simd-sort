/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512FP16_QSORT_16BIT
#define AVX512FP16_QSORT_16BIT

#include "avx512-16bit-common.h"

X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(_Float16 *arr,
                                                  int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask32 loadmask = 0xFFFFFFFF;
    __m512h in_zmm;
    while (arrsize > 0) {
        if (arrsize < 32) {
            loadmask = (0x00000001 << arrsize) - 0x00000001;
            in_zmm = _mm512_castsi512_ph(
                    _mm512_maskz_loadu_epi16(loadmask, arr));
        }
        else {
            in_zmm = _mm512_loadu_ph(arr);
        }
        __mmask32 nanmask = _mm512_cmp_ph_mask(in_zmm, in_zmm, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm512_mask_storeu_epi16(arr, nanmask, ZMM_MAX_HALF);
        arr += 32;
        arrsize -= 32;
    }
    return nan_count;
}

X86_SIMD_SORT_INLINE void
replace_inf_with_nan(_Float16 *arr, int64_t arrsize, int64_t nan_count)
{
    memset(arr + arrsize - nan_count, 0xFF, nan_count * 2);
}

template <>
void avx512_qselect(_Float16 *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qselect_16bit_<zmm_vector<_Float16>, _Float16>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

template <>
void avx512_qsort(_Float16 *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_16bit_<zmm_vector<_Float16>, _Float16>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // AVX512FP16_QSORT_16BIT
