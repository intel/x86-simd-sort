/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_QSORT_16BIT
#define AVX512_QSORT_16BIT

#include "avx512-16bit-common.h"

template <>
bool comparison_func<zmm_vector<float16>>(const uint16_t &a, const uint16_t &b)
{
    uint16_t signa = a & 0x8000, signb = b & 0x8000;
    uint16_t expa = a & 0x7c00, expb = b & 0x7c00;
    uint16_t manta = a & 0x3ff, mantb = b & 0x3ff;
    if (signa != signb) {
        // opposite signs
        return a > b;
    }
    else if (signa > 0) {
        // both -ve
        if (expa != expb) { return expa > expb; }
        else {
            return manta > mantb;
        }
    }
    else {
        // both +ve
        if (expa != expb) { return expa < expb; }
        else {
            return manta < mantb;
        }
    }

    //return npy_half_to_float(a) < npy_half_to_float(b);
}

X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(uint16_t *arr,
                                                  int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask16 loadmask = 0xFFFF;
    while (arrsize > 0) {
        if (arrsize < 16) { loadmask = (0x0001 << arrsize) - 0x0001; }
        __m256i in_zmm = _mm256_maskz_loadu_epi16(loadmask, arr);
        __m512 in_zmm_asfloat = _mm512_cvtph_ps(in_zmm);
        __mmask16 nanmask = _mm512_cmp_ps_mask(
                in_zmm_asfloat, in_zmm_asfloat, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm256_mask_storeu_epi16(arr, nanmask, YMM_MAX_HALF);
        arr += 16;
        arrsize -= 16;
    }
    return nan_count;
}

X86_SIMD_SORT_INLINE void
replace_inf_with_nan(uint16_t *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = 0xFFFF;
        nan_count -= 1;
    }
}

template <>
void avx512_qselect(int16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        qselect_16bit_<zmm_vector<int16_t>, int16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qselect(uint16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        qselect_16bit_<zmm_vector<uint16_t>, uint16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

void avx512_qselect_fp16(uint16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qselect_16bit_<zmm_vector<float16>, uint16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

template <>
void avx512_qsort(int16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<zmm_vector<int16_t>, int16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort(uint16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<zmm_vector<uint16_t>, uint16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

void avx512_qsort_fp16(uint16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_16bit_<zmm_vector<float16>, uint16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

#endif // AVX512_QSORT_16BIT
