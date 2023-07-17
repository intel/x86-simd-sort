/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX2_QSORT_32BIT
#define AVX2_QSORT_32BIT

#include "avx2-32bit-common.h"

// Assumes ymm is bitonic and performs a recursive half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t bitonic_merge_ymm_64bit(ymm_t ymm)
{
    
    const typename vtype::opmask_t oxAA = _mm256_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
    const typename vtype::opmask_t oxCC = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
    const typename vtype::opmask_t oxF0 = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0);

    // 1) half_cleaner[8]: compare 0-4, 1-5, 2-6, 3-7
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(_mm256_set_epi32(NETWORK_64BIT_4), ymm),
            oxF0);
    // 2) half_cleaner[4]
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(_mm256_set_epi32(NETWORK_64BIT_3), ymm),
            oxCC);
    // 3) half_cleaner[1]
    ymm = cmp_merge<vtype>(
            ymm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    return ymm;
}
// Assumes ymm1 and ymm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_two_ymm_64bit(ymm_t &ymm1, ymm_t &ymm2)
{
    const __m256i rev_index = _mm256_set_epi32(NETWORK_64BIT_2);
    // 1) First step of a merging network: coex of ymm1 and ymm2 reversed
    ymm2 = vtype::permutexvar(rev_index, ymm2);
    ymm_t ymm3 = vtype::min(ymm1, ymm2);
    ymm_t ymm4 = vtype::max(ymm1, ymm2);
    // 2) Recursive half cleaner for each
    ymm1 = bitonic_merge_ymm_64bit<vtype>(ymm3);
    ymm2 = bitonic_merge_ymm_64bit<vtype>(ymm4);
}
// Assumes [ymm0, ymm1] and [ymm2, ymm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_four_ymm_64bit(ymm_t *ymm)
{
    const __m256i rev_index = _mm256_set_epi32(NETWORK_64BIT_2);
    // 1) First step of a merging network
    ymm_t ymm2r = vtype::permutexvar(rev_index, ymm[2]);
    ymm_t ymm3r = vtype::permutexvar(rev_index, ymm[3]);
    ymm_t ymm_t1 = vtype::min(ymm[0], ymm3r);
    ymm_t ymm_t2 = vtype::min(ymm[1], ymm2r);
    // 2) Recursive half clearer: 16
    ymm_t ymm_t3 = vtype::permutexvar(rev_index, vtype::max(ymm[1], ymm2r));
    ymm_t ymm_t4 = vtype::permutexvar(rev_index, vtype::max(ymm[0], ymm3r));
    ymm_t ymm0 = vtype::min(ymm_t1, ymm_t2);
    ymm_t ymm1 = vtype::max(ymm_t1, ymm_t2);
    ymm_t ymm2 = vtype::min(ymm_t3, ymm_t4);
    ymm_t ymm3 = vtype::max(ymm_t3, ymm_t4);
    ymm[0] = bitonic_merge_ymm_64bit<vtype>(ymm0);
    ymm[1] = bitonic_merge_ymm_64bit<vtype>(ymm1);
    ymm[2] = bitonic_merge_ymm_64bit<vtype>(ymm2);
    ymm[3] = bitonic_merge_ymm_64bit<vtype>(ymm3);
}
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_eight_ymm_64bit(ymm_t *ymm)
{
    const __m256i rev_index = _mm256_set_epi32(NETWORK_64BIT_2);
    ymm_t ymm4r = vtype::permutexvar(rev_index, ymm[4]);
    ymm_t ymm5r = vtype::permutexvar(rev_index, ymm[5]);
    ymm_t ymm6r = vtype::permutexvar(rev_index, ymm[6]);
    ymm_t ymm7r = vtype::permutexvar(rev_index, ymm[7]);
    ymm_t ymm_t1 = vtype::min(ymm[0], ymm7r);
    ymm_t ymm_t2 = vtype::min(ymm[1], ymm6r);
    ymm_t ymm_t3 = vtype::min(ymm[2], ymm5r);
    ymm_t ymm_t4 = vtype::min(ymm[3], ymm4r);
    ymm_t ymm_t5 = vtype::permutexvar(rev_index, vtype::max(ymm[3], ymm4r));
    ymm_t ymm_t6 = vtype::permutexvar(rev_index, vtype::max(ymm[2], ymm5r));
    ymm_t ymm_t7 = vtype::permutexvar(rev_index, vtype::max(ymm[1], ymm6r));
    ymm_t ymm_t8 = vtype::permutexvar(rev_index, vtype::max(ymm[0], ymm7r));
    COEX<vtype>(ymm_t1, ymm_t3);
    COEX<vtype>(ymm_t2, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t7);
    COEX<vtype>(ymm_t6, ymm_t8);
    COEX<vtype>(ymm_t1, ymm_t2);
    COEX<vtype>(ymm_t3, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t6);
    COEX<vtype>(ymm_t7, ymm_t8);
    ymm[0] = bitonic_merge_ymm_64bit<vtype>(ymm_t1);
    ymm[1] = bitonic_merge_ymm_64bit<vtype>(ymm_t2);
    ymm[2] = bitonic_merge_ymm_64bit<vtype>(ymm_t3);
    ymm[3] = bitonic_merge_ymm_64bit<vtype>(ymm_t4);
    ymm[4] = bitonic_merge_ymm_64bit<vtype>(ymm_t5);
    ymm[5] = bitonic_merge_ymm_64bit<vtype>(ymm_t6);
    ymm[6] = bitonic_merge_ymm_64bit<vtype>(ymm_t7);
    ymm[7] = bitonic_merge_ymm_64bit<vtype>(ymm_t8);
}
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_sixteen_ymm_64bit(ymm_t *ymm)
{
    const __m256i rev_index = _mm256_set_epi32(NETWORK_64BIT_2);
    ymm_t ymm8r = vtype::permutexvar(rev_index, ymm[8]);
    ymm_t ymm9r = vtype::permutexvar(rev_index, ymm[9]);
    ymm_t ymm10r = vtype::permutexvar(rev_index, ymm[10]);
    ymm_t ymm11r = vtype::permutexvar(rev_index, ymm[11]);
    ymm_t ymm12r = vtype::permutexvar(rev_index, ymm[12]);
    ymm_t ymm13r = vtype::permutexvar(rev_index, ymm[13]);
    ymm_t ymm14r = vtype::permutexvar(rev_index, ymm[14]);
    ymm_t ymm15r = vtype::permutexvar(rev_index, ymm[15]);
    ymm_t ymm_t1 = vtype::min(ymm[0], ymm15r);
    ymm_t ymm_t2 = vtype::min(ymm[1], ymm14r);
    ymm_t ymm_t3 = vtype::min(ymm[2], ymm13r);
    ymm_t ymm_t4 = vtype::min(ymm[3], ymm12r);
    ymm_t ymm_t5 = vtype::min(ymm[4], ymm11r);
    ymm_t ymm_t6 = vtype::min(ymm[5], ymm10r);
    ymm_t ymm_t7 = vtype::min(ymm[6], ymm9r);
    ymm_t ymm_t8 = vtype::min(ymm[7], ymm8r);
    ymm_t ymm_t9 = vtype::permutexvar(rev_index, vtype::max(ymm[7], ymm8r));
    ymm_t ymm_t10 = vtype::permutexvar(rev_index, vtype::max(ymm[6], ymm9r));
    ymm_t ymm_t11 = vtype::permutexvar(rev_index, vtype::max(ymm[5], ymm10r));
    ymm_t ymm_t12 = vtype::permutexvar(rev_index, vtype::max(ymm[4], ymm11r));
    ymm_t ymm_t13 = vtype::permutexvar(rev_index, vtype::max(ymm[3], ymm12r));
    ymm_t ymm_t14 = vtype::permutexvar(rev_index, vtype::max(ymm[2], ymm13r));
    ymm_t ymm_t15 = vtype::permutexvar(rev_index, vtype::max(ymm[1], ymm14r));
    ymm_t ymm_t16 = vtype::permutexvar(rev_index, vtype::max(ymm[0], ymm15r));
    // Recusive half clear 16 ymm regs
    COEX<vtype>(ymm_t1, ymm_t5);
    COEX<vtype>(ymm_t2, ymm_t6);
    COEX<vtype>(ymm_t3, ymm_t7);
    COEX<vtype>(ymm_t4, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t13);
    COEX<vtype>(ymm_t10, ymm_t14);
    COEX<vtype>(ymm_t11, ymm_t15);
    COEX<vtype>(ymm_t12, ymm_t16);
    //
    COEX<vtype>(ymm_t1, ymm_t3);
    COEX<vtype>(ymm_t2, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t7);
    COEX<vtype>(ymm_t6, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t11);
    COEX<vtype>(ymm_t10, ymm_t12);
    COEX<vtype>(ymm_t13, ymm_t15);
    COEX<vtype>(ymm_t14, ymm_t16);
    //
    COEX<vtype>(ymm_t1, ymm_t2);
    COEX<vtype>(ymm_t3, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t6);
    COEX<vtype>(ymm_t7, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t10);
    COEX<vtype>(ymm_t11, ymm_t12);
    COEX<vtype>(ymm_t13, ymm_t14);
    COEX<vtype>(ymm_t15, ymm_t16);
    //
    ymm[0] = bitonic_merge_ymm_64bit<vtype>(ymm_t1);
    ymm[1] = bitonic_merge_ymm_64bit<vtype>(ymm_t2);
    ymm[2] = bitonic_merge_ymm_64bit<vtype>(ymm_t3);
    ymm[3] = bitonic_merge_ymm_64bit<vtype>(ymm_t4);
    ymm[4] = bitonic_merge_ymm_64bit<vtype>(ymm_t5);
    ymm[5] = bitonic_merge_ymm_64bit<vtype>(ymm_t6);
    ymm[6] = bitonic_merge_ymm_64bit<vtype>(ymm_t7);
    ymm[7] = bitonic_merge_ymm_64bit<vtype>(ymm_t8);
    ymm[8] = bitonic_merge_ymm_64bit<vtype>(ymm_t9);
    ymm[9] = bitonic_merge_ymm_64bit<vtype>(ymm_t10);
    ymm[10] = bitonic_merge_ymm_64bit<vtype>(ymm_t11);
    ymm[11] = bitonic_merge_ymm_64bit<vtype>(ymm_t12);
    ymm[12] = bitonic_merge_ymm_64bit<vtype>(ymm_t13);
    ymm[13] = bitonic_merge_ymm_64bit<vtype>(ymm_t14);
    ymm[14] = bitonic_merge_ymm_64bit<vtype>(ymm_t15);
    ymm[15] = bitonic_merge_ymm_64bit<vtype>(ymm_t16);
}

template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_32_ymm_64bit(ymm_t *ymm)
{
    const __m256i rev_index = _mm256_set_epi32(NETWORK_64BIT_2);
    ymm_t ymm16r = vtype::permutexvar(rev_index, ymm[16]);
    ymm_t ymm17r = vtype::permutexvar(rev_index, ymm[17]);
    ymm_t ymm18r = vtype::permutexvar(rev_index, ymm[18]);
    ymm_t ymm19r = vtype::permutexvar(rev_index, ymm[19]);
    ymm_t ymm20r = vtype::permutexvar(rev_index, ymm[20]);
    ymm_t ymm21r = vtype::permutexvar(rev_index, ymm[21]);
    ymm_t ymm22r = vtype::permutexvar(rev_index, ymm[22]);
    ymm_t ymm23r = vtype::permutexvar(rev_index, ymm[23]);
    ymm_t ymm24r = vtype::permutexvar(rev_index, ymm[24]);
    ymm_t ymm25r = vtype::permutexvar(rev_index, ymm[25]);
    ymm_t ymm26r = vtype::permutexvar(rev_index, ymm[26]);
    ymm_t ymm27r = vtype::permutexvar(rev_index, ymm[27]);
    ymm_t ymm28r = vtype::permutexvar(rev_index, ymm[28]);
    ymm_t ymm29r = vtype::permutexvar(rev_index, ymm[29]);
    ymm_t ymm30r = vtype::permutexvar(rev_index, ymm[30]);
    ymm_t ymm31r = vtype::permutexvar(rev_index, ymm[31]);
    ymm_t ymm_t1 = vtype::min(ymm[0], ymm31r);
    ymm_t ymm_t2 = vtype::min(ymm[1], ymm30r);
    ymm_t ymm_t3 = vtype::min(ymm[2], ymm29r);
    ymm_t ymm_t4 = vtype::min(ymm[3], ymm28r);
    ymm_t ymm_t5 = vtype::min(ymm[4], ymm27r);
    ymm_t ymm_t6 = vtype::min(ymm[5], ymm26r);
    ymm_t ymm_t7 = vtype::min(ymm[6], ymm25r);
    ymm_t ymm_t8 = vtype::min(ymm[7], ymm24r);
    ymm_t ymm_t9 = vtype::min(ymm[8], ymm23r);
    ymm_t ymm_t10 = vtype::min(ymm[9], ymm22r);
    ymm_t ymm_t11 = vtype::min(ymm[10], ymm21r);
    ymm_t ymm_t12 = vtype::min(ymm[11], ymm20r);
    ymm_t ymm_t13 = vtype::min(ymm[12], ymm19r);
    ymm_t ymm_t14 = vtype::min(ymm[13], ymm18r);
    ymm_t ymm_t15 = vtype::min(ymm[14], ymm17r);
    ymm_t ymm_t16 = vtype::min(ymm[15], ymm16r);
    ymm_t ymm_t17 = vtype::permutexvar(rev_index, vtype::max(ymm[15], ymm16r));
    ymm_t ymm_t18 = vtype::permutexvar(rev_index, vtype::max(ymm[14], ymm17r));
    ymm_t ymm_t19 = vtype::permutexvar(rev_index, vtype::max(ymm[13], ymm18r));
    ymm_t ymm_t20 = vtype::permutexvar(rev_index, vtype::max(ymm[12], ymm19r));
    ymm_t ymm_t21 = vtype::permutexvar(rev_index, vtype::max(ymm[11], ymm20r));
    ymm_t ymm_t22 = vtype::permutexvar(rev_index, vtype::max(ymm[10], ymm21r));
    ymm_t ymm_t23 = vtype::permutexvar(rev_index, vtype::max(ymm[9], ymm22r));
    ymm_t ymm_t24 = vtype::permutexvar(rev_index, vtype::max(ymm[8], ymm23r));
    ymm_t ymm_t25 = vtype::permutexvar(rev_index, vtype::max(ymm[7], ymm24r));
    ymm_t ymm_t26 = vtype::permutexvar(rev_index, vtype::max(ymm[6], ymm25r));
    ymm_t ymm_t27 = vtype::permutexvar(rev_index, vtype::max(ymm[5], ymm26r));
    ymm_t ymm_t28 = vtype::permutexvar(rev_index, vtype::max(ymm[4], ymm27r));
    ymm_t ymm_t29 = vtype::permutexvar(rev_index, vtype::max(ymm[3], ymm28r));
    ymm_t ymm_t30 = vtype::permutexvar(rev_index, vtype::max(ymm[2], ymm29r));
    ymm_t ymm_t31 = vtype::permutexvar(rev_index, vtype::max(ymm[1], ymm30r));
    ymm_t ymm_t32 = vtype::permutexvar(rev_index, vtype::max(ymm[0], ymm31r));
    // Recusive half clear 16 ymm regs
    COEX<vtype>(ymm_t1, ymm_t9);
    COEX<vtype>(ymm_t2, ymm_t10);
    COEX<vtype>(ymm_t3, ymm_t11);
    COEX<vtype>(ymm_t4, ymm_t12);
    COEX<vtype>(ymm_t5, ymm_t13);
    COEX<vtype>(ymm_t6, ymm_t14);
    COEX<vtype>(ymm_t7, ymm_t15);
    COEX<vtype>(ymm_t8, ymm_t16);
    COEX<vtype>(ymm_t17, ymm_t25);
    COEX<vtype>(ymm_t18, ymm_t26);
    COEX<vtype>(ymm_t19, ymm_t27);
    COEX<vtype>(ymm_t20, ymm_t28);
    COEX<vtype>(ymm_t21, ymm_t29);
    COEX<vtype>(ymm_t22, ymm_t30);
    COEX<vtype>(ymm_t23, ymm_t31);
    COEX<vtype>(ymm_t24, ymm_t32);
    //
    COEX<vtype>(ymm_t1, ymm_t5);
    COEX<vtype>(ymm_t2, ymm_t6);
    COEX<vtype>(ymm_t3, ymm_t7);
    COEX<vtype>(ymm_t4, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t13);
    COEX<vtype>(ymm_t10, ymm_t14);
    COEX<vtype>(ymm_t11, ymm_t15);
    COEX<vtype>(ymm_t12, ymm_t16);
    COEX<vtype>(ymm_t17, ymm_t21);
    COEX<vtype>(ymm_t18, ymm_t22);
    COEX<vtype>(ymm_t19, ymm_t23);
    COEX<vtype>(ymm_t20, ymm_t24);
    COEX<vtype>(ymm_t25, ymm_t29);
    COEX<vtype>(ymm_t26, ymm_t30);
    COEX<vtype>(ymm_t27, ymm_t31);
    COEX<vtype>(ymm_t28, ymm_t32);
    //
    COEX<vtype>(ymm_t1, ymm_t3);
    COEX<vtype>(ymm_t2, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t7);
    COEX<vtype>(ymm_t6, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t11);
    COEX<vtype>(ymm_t10, ymm_t12);
    COEX<vtype>(ymm_t13, ymm_t15);
    COEX<vtype>(ymm_t14, ymm_t16);
    COEX<vtype>(ymm_t17, ymm_t19);
    COEX<vtype>(ymm_t18, ymm_t20);
    COEX<vtype>(ymm_t21, ymm_t23);
    COEX<vtype>(ymm_t22, ymm_t24);
    COEX<vtype>(ymm_t25, ymm_t27);
    COEX<vtype>(ymm_t26, ymm_t28);
    COEX<vtype>(ymm_t29, ymm_t31);
    COEX<vtype>(ymm_t30, ymm_t32);
    //
    COEX<vtype>(ymm_t1, ymm_t2);
    COEX<vtype>(ymm_t3, ymm_t4);
    COEX<vtype>(ymm_t5, ymm_t6);
    COEX<vtype>(ymm_t7, ymm_t8);
    COEX<vtype>(ymm_t9, ymm_t10);
    COEX<vtype>(ymm_t11, ymm_t12);
    COEX<vtype>(ymm_t13, ymm_t14);
    COEX<vtype>(ymm_t15, ymm_t16);
    COEX<vtype>(ymm_t17, ymm_t18);
    COEX<vtype>(ymm_t19, ymm_t20);
    COEX<vtype>(ymm_t21, ymm_t22);
    COEX<vtype>(ymm_t23, ymm_t24);
    COEX<vtype>(ymm_t25, ymm_t26);
    COEX<vtype>(ymm_t27, ymm_t28);
    COEX<vtype>(ymm_t29, ymm_t30);
    COEX<vtype>(ymm_t31, ymm_t32);
    //
    ymm[0] = bitonic_merge_ymm_64bit<vtype>(ymm_t1);
    ymm[1] = bitonic_merge_ymm_64bit<vtype>(ymm_t2);
    ymm[2] = bitonic_merge_ymm_64bit<vtype>(ymm_t3);
    ymm[3] = bitonic_merge_ymm_64bit<vtype>(ymm_t4);
    ymm[4] = bitonic_merge_ymm_64bit<vtype>(ymm_t5);
    ymm[5] = bitonic_merge_ymm_64bit<vtype>(ymm_t6);
    ymm[6] = bitonic_merge_ymm_64bit<vtype>(ymm_t7);
    ymm[7] = bitonic_merge_ymm_64bit<vtype>(ymm_t8);
    ymm[8] = bitonic_merge_ymm_64bit<vtype>(ymm_t9);
    ymm[9] = bitonic_merge_ymm_64bit<vtype>(ymm_t10);
    ymm[10] = bitonic_merge_ymm_64bit<vtype>(ymm_t11);
    ymm[11] = bitonic_merge_ymm_64bit<vtype>(ymm_t12);
    ymm[12] = bitonic_merge_ymm_64bit<vtype>(ymm_t13);
    ymm[13] = bitonic_merge_ymm_64bit<vtype>(ymm_t14);
    ymm[14] = bitonic_merge_ymm_64bit<vtype>(ymm_t15);
    ymm[15] = bitonic_merge_ymm_64bit<vtype>(ymm_t16);
    ymm[16] = bitonic_merge_ymm_64bit<vtype>(ymm_t17);
    ymm[17] = bitonic_merge_ymm_64bit<vtype>(ymm_t18);
    ymm[18] = bitonic_merge_ymm_64bit<vtype>(ymm_t19);
    ymm[19] = bitonic_merge_ymm_64bit<vtype>(ymm_t20);
    ymm[20] = bitonic_merge_ymm_64bit<vtype>(ymm_t21);
    ymm[21] = bitonic_merge_ymm_64bit<vtype>(ymm_t22);
    ymm[22] = bitonic_merge_ymm_64bit<vtype>(ymm_t23);
    ymm[23] = bitonic_merge_ymm_64bit<vtype>(ymm_t24);
    ymm[24] = bitonic_merge_ymm_64bit<vtype>(ymm_t25);
    ymm[25] = bitonic_merge_ymm_64bit<vtype>(ymm_t26);
    ymm[26] = bitonic_merge_ymm_64bit<vtype>(ymm_t27);
    ymm[27] = bitonic_merge_ymm_64bit<vtype>(ymm_t28);
    ymm[28] = bitonic_merge_ymm_64bit<vtype>(ymm_t29);
    ymm[29] = bitonic_merge_ymm_64bit<vtype>(ymm_t30);
    ymm[30] = bitonic_merge_ymm_64bit<vtype>(ymm_t31);
    ymm[31] = bitonic_merge_ymm_64bit<vtype>(ymm_t32);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_8_64bit(type_t *arr, int32_t N)
{
    typename vtype::opmask_t load_mask = (0x01 << N) - 0x01;
    typename vtype::ymm_t ymm
            = vtype::mask_loadu(vtype::ymm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_ymm_64bit<vtype>(ymm));
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_16_64bit(type_t *arr, int32_t N)
{
    if (N <= 8) {
        sort_8_64bit<vtype>(arr, N);
        return;
    }
    using ymm_t = typename vtype::ymm_t;
    ymm_t ymm1 = vtype::loadu(arr);
    typename vtype::opmask_t load_mask = (0x01 << (N - 8)) - 0x01;
    ymm_t ymm2 = vtype::mask_loadu(vtype::ymm_max(), load_mask, arr + 8);
    ymm1 = sort_ymm_64bit<vtype>(ymm1);
    ymm2 = sort_ymm_64bit<vtype>(ymm2);
    bitonic_merge_two_ymm_64bit<vtype>(ymm1, ymm2);
    vtype::storeu(arr, ymm1);
    vtype::mask_storeu(arr + 8, load_mask, ymm2);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_32_64bit(type_t *arr, int32_t N)
{
    if (N <= 16) {
        sort_16_64bit<vtype>(arr, N);
        return;
    }
    using ymm_t = typename vtype::ymm_t;
    using opmask_t = typename vtype::opmask_t;
    ymm_t ymm[4];
    ymm[0] = vtype::loadu(arr);
    ymm[1] = vtype::loadu(arr + 8);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    uint64_t combined_mask = (0x1ull << (N - 16)) - 0x1ull;
    load_mask1 = (combined_mask)&0xFF;
    load_mask2 = (combined_mask >> 8) & 0xFF;
    ymm[2] = vtype::mask_loadu(vtype::ymm_max(), load_mask1, arr + 16);
    ymm[3] = vtype::mask_loadu(vtype::ymm_max(), load_mask2, arr + 24);
    ymm[0] = sort_ymm_64bit<vtype>(ymm[0]);
    ymm[1] = sort_ymm_64bit<vtype>(ymm[1]);
    ymm[2] = sort_ymm_64bit<vtype>(ymm[2]);
    ymm[3] = sort_ymm_64bit<vtype>(ymm[3]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[0], ymm[1]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[2], ymm[3]);
    bitonic_merge_four_ymm_64bit<vtype>(ymm);
    vtype::storeu(arr, ymm[0]);
    vtype::storeu(arr + 8, ymm[1]);
    vtype::mask_storeu(arr + 16, load_mask1, ymm[2]);
    vtype::mask_storeu(arr + 24, load_mask2, ymm[3]);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_64_64bit(type_t *arr, int32_t N)
{
    if (N <= 32) {
        sort_32_64bit<vtype>(arr, N);
        return;
    }
    using ymm_t = typename vtype::ymm_t;
    using opmask_t = typename vtype::opmask_t;
    ymm_t ymm[8];
    ymm[0] = vtype::loadu(arr);
    ymm[1] = vtype::loadu(arr + 8);
    ymm[2] = vtype::loadu(arr + 16);
    ymm[3] = vtype::loadu(arr + 24);
    ymm[0] = sort_ymm_64bit<vtype>(ymm[0]);
    ymm[1] = sort_ymm_64bit<vtype>(ymm[1]);
    ymm[2] = sort_ymm_64bit<vtype>(ymm[2]);
    ymm[3] = sort_ymm_64bit<vtype>(ymm[3]);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    opmask_t load_mask3 = 0xFF, load_mask4 = 0xFF;
    // N-32 >= 1
    uint64_t combined_mask = (0x1ull << (N - 32)) - 0x1ull;
    load_mask1 = (combined_mask)&0xFF;
    load_mask2 = (combined_mask >> 8) & 0xFF;
    load_mask3 = (combined_mask >> 16) & 0xFF;
    load_mask4 = (combined_mask >> 24) & 0xFF;
    ymm[4] = vtype::mask_loadu(vtype::ymm_max(), load_mask1, arr + 32);
    ymm[5] = vtype::mask_loadu(vtype::ymm_max(), load_mask2, arr + 40);
    ymm[6] = vtype::mask_loadu(vtype::ymm_max(), load_mask3, arr + 48);
    ymm[7] = vtype::mask_loadu(vtype::ymm_max(), load_mask4, arr + 56);
    ymm[4] = sort_ymm_64bit<vtype>(ymm[4]);
    ymm[5] = sort_ymm_64bit<vtype>(ymm[5]);
    ymm[6] = sort_ymm_64bit<vtype>(ymm[6]);
    ymm[7] = sort_ymm_64bit<vtype>(ymm[7]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[0], ymm[1]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[2], ymm[3]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[4], ymm[5]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[6], ymm[7]);
    bitonic_merge_four_ymm_64bit<vtype>(ymm);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 4);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm);
    vtype::storeu(arr, ymm[0]);
    vtype::storeu(arr + 8, ymm[1]);
    vtype::storeu(arr + 16, ymm[2]);
    vtype::storeu(arr + 24, ymm[3]);
    vtype::mask_storeu(arr + 32, load_mask1, ymm[4]);
    vtype::mask_storeu(arr + 40, load_mask2, ymm[5]);
    vtype::mask_storeu(arr + 48, load_mask3, ymm[6]);
    vtype::mask_storeu(arr + 56, load_mask4, ymm[7]);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_128_64bit(type_t *arr, int32_t N)
{
    if (N <= 64) {
        sort_64_64bit<vtype>(arr, N);
        return;
    }
    using ymm_t = typename vtype::ymm_t;
    using opmask_t = typename vtype::opmask_t;
    ymm_t ymm[16];
    ymm[0] = vtype::loadu(arr);
    ymm[1] = vtype::loadu(arr + 8);
    ymm[2] = vtype::loadu(arr + 16);
    ymm[3] = vtype::loadu(arr + 24);
    ymm[4] = vtype::loadu(arr + 32);
    ymm[5] = vtype::loadu(arr + 40);
    ymm[6] = vtype::loadu(arr + 48);
    ymm[7] = vtype::loadu(arr + 56);
    ymm[0] = sort_ymm_64bit<vtype>(ymm[0]);
    ymm[1] = sort_ymm_64bit<vtype>(ymm[1]);
    ymm[2] = sort_ymm_64bit<vtype>(ymm[2]);
    ymm[3] = sort_ymm_64bit<vtype>(ymm[3]);
    ymm[4] = sort_ymm_64bit<vtype>(ymm[4]);
    ymm[5] = sort_ymm_64bit<vtype>(ymm[5]);
    ymm[6] = sort_ymm_64bit<vtype>(ymm[6]);
    ymm[7] = sort_ymm_64bit<vtype>(ymm[7]);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    opmask_t load_mask3 = 0xFF, load_mask4 = 0xFF;
    opmask_t load_mask5 = 0xFF, load_mask6 = 0xFF;
    opmask_t load_mask7 = 0xFF, load_mask8 = 0xFF;
    if (N != 128) {
        uint64_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
        load_mask1 = (combined_mask)&0xFF;
        load_mask2 = (combined_mask >> 8) & 0xFF;
        load_mask3 = (combined_mask >> 16) & 0xFF;
        load_mask4 = (combined_mask >> 24) & 0xFF;
        load_mask5 = (combined_mask >> 32) & 0xFF;
        load_mask6 = (combined_mask >> 40) & 0xFF;
        load_mask7 = (combined_mask >> 48) & 0xFF;
        load_mask8 = (combined_mask >> 56) & 0xFF;
    }
    ymm[8] = vtype::mask_loadu(vtype::ymm_max(), load_mask1, arr + 64);
    ymm[9] = vtype::mask_loadu(vtype::ymm_max(), load_mask2, arr + 72);
    ymm[10] = vtype::mask_loadu(vtype::ymm_max(), load_mask3, arr + 80);
    ymm[11] = vtype::mask_loadu(vtype::ymm_max(), load_mask4, arr + 88);
    ymm[12] = vtype::mask_loadu(vtype::ymm_max(), load_mask5, arr + 96);
    ymm[13] = vtype::mask_loadu(vtype::ymm_max(), load_mask6, arr + 104);
    ymm[14] = vtype::mask_loadu(vtype::ymm_max(), load_mask7, arr + 112);
    ymm[15] = vtype::mask_loadu(vtype::ymm_max(), load_mask8, arr + 120);
    ymm[8] = sort_ymm_64bit<vtype>(ymm[8]);
    ymm[9] = sort_ymm_64bit<vtype>(ymm[9]);
    ymm[10] = sort_ymm_64bit<vtype>(ymm[10]);
    ymm[11] = sort_ymm_64bit<vtype>(ymm[11]);
    ymm[12] = sort_ymm_64bit<vtype>(ymm[12]);
    ymm[13] = sort_ymm_64bit<vtype>(ymm[13]);
    ymm[14] = sort_ymm_64bit<vtype>(ymm[14]);
    ymm[15] = sort_ymm_64bit<vtype>(ymm[15]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[0], ymm[1]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[2], ymm[3]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[4], ymm[5]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[6], ymm[7]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[8], ymm[9]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[10], ymm[11]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[12], ymm[13]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[14], ymm[15]);
    bitonic_merge_four_ymm_64bit<vtype>(ymm);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 4);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 8);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 12);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm + 8);
    bitonic_merge_sixteen_ymm_64bit<vtype>(ymm);
    vtype::storeu(arr, ymm[0]);
    vtype::storeu(arr + 8, ymm[1]);
    vtype::storeu(arr + 16, ymm[2]);
    vtype::storeu(arr + 24, ymm[3]);
    vtype::storeu(arr + 32, ymm[4]);
    vtype::storeu(arr + 40, ymm[5]);
    vtype::storeu(arr + 48, ymm[6]);
    vtype::storeu(arr + 56, ymm[7]);
    vtype::mask_storeu(arr + 64, load_mask1, ymm[8]);
    vtype::mask_storeu(arr + 72, load_mask2, ymm[9]);
    vtype::mask_storeu(arr + 80, load_mask3, ymm[10]);
    vtype::mask_storeu(arr + 88, load_mask4, ymm[11]);
    vtype::mask_storeu(arr + 96, load_mask5, ymm[12]);
    vtype::mask_storeu(arr + 104, load_mask6, ymm[13]);
    vtype::mask_storeu(arr + 112, load_mask7, ymm[14]);
    vtype::mask_storeu(arr + 120, load_mask8, ymm[15]);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_256_64bit(type_t *arr, int32_t N)
{
    if (N <= 128) {
        sort_128_64bit<vtype>(arr, N);
        return;
    }
    using ymm_t = typename vtype::ymm_t;
    using opmask_t = typename vtype::opmask_t;
    ymm_t ymm[32];
    ymm[0] = vtype::loadu(arr);
    ymm[1] = vtype::loadu(arr + 8);
    ymm[2] = vtype::loadu(arr + 16);
    ymm[3] = vtype::loadu(arr + 24);
    ymm[4] = vtype::loadu(arr + 32);
    ymm[5] = vtype::loadu(arr + 40);
    ymm[6] = vtype::loadu(arr + 48);
    ymm[7] = vtype::loadu(arr + 56);
    ymm[8] = vtype::loadu(arr + 64);
    ymm[9] = vtype::loadu(arr + 72);
    ymm[10] = vtype::loadu(arr + 80);
    ymm[11] = vtype::loadu(arr + 88);
    ymm[12] = vtype::loadu(arr + 96);
    ymm[13] = vtype::loadu(arr + 104);
    ymm[14] = vtype::loadu(arr + 112);
    ymm[15] = vtype::loadu(arr + 120);
    ymm[0] = sort_ymm_64bit<vtype>(ymm[0]);
    ymm[1] = sort_ymm_64bit<vtype>(ymm[1]);
    ymm[2] = sort_ymm_64bit<vtype>(ymm[2]);
    ymm[3] = sort_ymm_64bit<vtype>(ymm[3]);
    ymm[4] = sort_ymm_64bit<vtype>(ymm[4]);
    ymm[5] = sort_ymm_64bit<vtype>(ymm[5]);
    ymm[6] = sort_ymm_64bit<vtype>(ymm[6]);
    ymm[7] = sort_ymm_64bit<vtype>(ymm[7]);
    ymm[8] = sort_ymm_64bit<vtype>(ymm[8]);
    ymm[9] = sort_ymm_64bit<vtype>(ymm[9]);
    ymm[10] = sort_ymm_64bit<vtype>(ymm[10]);
    ymm[11] = sort_ymm_64bit<vtype>(ymm[11]);
    ymm[12] = sort_ymm_64bit<vtype>(ymm[12]);
    ymm[13] = sort_ymm_64bit<vtype>(ymm[13]);
    ymm[14] = sort_ymm_64bit<vtype>(ymm[14]);
    ymm[15] = sort_ymm_64bit<vtype>(ymm[15]);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    opmask_t load_mask3 = 0xFF, load_mask4 = 0xFF;
    opmask_t load_mask5 = 0xFF, load_mask6 = 0xFF;
    opmask_t load_mask7 = 0xFF, load_mask8 = 0xFF;
    opmask_t load_mask9 = 0xFF, load_mask10 = 0xFF;
    opmask_t load_mask11 = 0xFF, load_mask12 = 0xFF;
    opmask_t load_mask13 = 0xFF, load_mask14 = 0xFF;
    opmask_t load_mask15 = 0xFF, load_mask16 = 0xFF;
    if (N != 256) {
        uint64_t combined_mask;
        if (N < 192) {
            combined_mask = (0x1ull << (N - 128)) - 0x1ull;
            load_mask1 = (combined_mask)&0xFF;
            load_mask2 = (combined_mask >> 8) & 0xFF;
            load_mask3 = (combined_mask >> 16) & 0xFF;
            load_mask4 = (combined_mask >> 24) & 0xFF;
            load_mask5 = (combined_mask >> 32) & 0xFF;
            load_mask6 = (combined_mask >> 40) & 0xFF;
            load_mask7 = (combined_mask >> 48) & 0xFF;
            load_mask8 = (combined_mask >> 56) & 0xFF;
            load_mask9 = 0x00;
            load_mask10 = 0x0;
            load_mask11 = 0x00;
            load_mask12 = 0x00;
            load_mask13 = 0x00;
            load_mask14 = 0x00;
            load_mask15 = 0x00;
            load_mask16 = 0x00;
        }
        else {
            combined_mask = (0x1ull << (N - 192)) - 0x1ull;
            load_mask9 = (combined_mask)&0xFF;
            load_mask10 = (combined_mask >> 8) & 0xFF;
            load_mask11 = (combined_mask >> 16) & 0xFF;
            load_mask12 = (combined_mask >> 24) & 0xFF;
            load_mask13 = (combined_mask >> 32) & 0xFF;
            load_mask14 = (combined_mask >> 40) & 0xFF;
            load_mask15 = (combined_mask >> 48) & 0xFF;
            load_mask16 = (combined_mask >> 56) & 0xFF;
        }
    }
    ymm[16] = vtype::mask_loadu(vtype::ymm_max(), load_mask1, arr + 128);
    ymm[17] = vtype::mask_loadu(vtype::ymm_max(), load_mask2, arr + 136);
    ymm[18] = vtype::mask_loadu(vtype::ymm_max(), load_mask3, arr + 144);
    ymm[19] = vtype::mask_loadu(vtype::ymm_max(), load_mask4, arr + 152);
    ymm[20] = vtype::mask_loadu(vtype::ymm_max(), load_mask5, arr + 160);
    ymm[21] = vtype::mask_loadu(vtype::ymm_max(), load_mask6, arr + 168);
    ymm[22] = vtype::mask_loadu(vtype::ymm_max(), load_mask7, arr + 176);
    ymm[23] = vtype::mask_loadu(vtype::ymm_max(), load_mask8, arr + 184);
    if (N < 192) {
        ymm[24] = vtype::ymm_max();
        ymm[25] = vtype::ymm_max();
        ymm[26] = vtype::ymm_max();
        ymm[27] = vtype::ymm_max();
        ymm[28] = vtype::ymm_max();
        ymm[29] = vtype::ymm_max();
        ymm[30] = vtype::ymm_max();
        ymm[31] = vtype::ymm_max();
    }
    else {
        ymm[24] = vtype::mask_loadu(vtype::ymm_max(), load_mask9, arr + 192);
        ymm[25] = vtype::mask_loadu(vtype::ymm_max(), load_mask10, arr + 200);
        ymm[26] = vtype::mask_loadu(vtype::ymm_max(), load_mask11, arr + 208);
        ymm[27] = vtype::mask_loadu(vtype::ymm_max(), load_mask12, arr + 216);
        ymm[28] = vtype::mask_loadu(vtype::ymm_max(), load_mask13, arr + 224);
        ymm[29] = vtype::mask_loadu(vtype::ymm_max(), load_mask14, arr + 232);
        ymm[30] = vtype::mask_loadu(vtype::ymm_max(), load_mask15, arr + 240);
        ymm[31] = vtype::mask_loadu(vtype::ymm_max(), load_mask16, arr + 248);
    }
    ymm[16] = sort_ymm_64bit<vtype>(ymm[16]);
    ymm[17] = sort_ymm_64bit<vtype>(ymm[17]);
    ymm[18] = sort_ymm_64bit<vtype>(ymm[18]);
    ymm[19] = sort_ymm_64bit<vtype>(ymm[19]);
    ymm[20] = sort_ymm_64bit<vtype>(ymm[20]);
    ymm[21] = sort_ymm_64bit<vtype>(ymm[21]);
    ymm[22] = sort_ymm_64bit<vtype>(ymm[22]);
    ymm[23] = sort_ymm_64bit<vtype>(ymm[23]);
    ymm[24] = sort_ymm_64bit<vtype>(ymm[24]);
    ymm[25] = sort_ymm_64bit<vtype>(ymm[25]);
    ymm[26] = sort_ymm_64bit<vtype>(ymm[26]);
    ymm[27] = sort_ymm_64bit<vtype>(ymm[27]);
    ymm[28] = sort_ymm_64bit<vtype>(ymm[28]);
    ymm[29] = sort_ymm_64bit<vtype>(ymm[29]);
    ymm[30] = sort_ymm_64bit<vtype>(ymm[30]);
    ymm[31] = sort_ymm_64bit<vtype>(ymm[31]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[0], ymm[1]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[2], ymm[3]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[4], ymm[5]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[6], ymm[7]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[8], ymm[9]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[10], ymm[11]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[12], ymm[13]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[14], ymm[15]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[16], ymm[17]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[18], ymm[19]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[20], ymm[21]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[22], ymm[23]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[24], ymm[25]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[26], ymm[27]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[28], ymm[29]);
    bitonic_merge_two_ymm_64bit<vtype>(ymm[30], ymm[31]);
    bitonic_merge_four_ymm_64bit<vtype>(ymm);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 4);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 8);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 12);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 16);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 20);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 24);
    bitonic_merge_four_ymm_64bit<vtype>(ymm + 28);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm + 8);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm + 16);
    bitonic_merge_eight_ymm_64bit<vtype>(ymm + 24);
    bitonic_merge_sixteen_ymm_64bit<vtype>(ymm);
    bitonic_merge_sixteen_ymm_64bit<vtype>(ymm + 16);
    bitonic_merge_32_ymm_64bit<vtype>(ymm);
    vtype::storeu(arr, ymm[0]);
    vtype::storeu(arr + 8, ymm[1]);
    vtype::storeu(arr + 16, ymm[2]);
    vtype::storeu(arr + 24, ymm[3]);
    vtype::storeu(arr + 32, ymm[4]);
    vtype::storeu(arr + 40, ymm[5]);
    vtype::storeu(arr + 48, ymm[6]);
    vtype::storeu(arr + 56, ymm[7]);
    vtype::storeu(arr + 64, ymm[8]);
    vtype::storeu(arr + 72, ymm[9]);
    vtype::storeu(arr + 80, ymm[10]);
    vtype::storeu(arr + 88, ymm[11]);
    vtype::storeu(arr + 96, ymm[12]);
    vtype::storeu(arr + 104, ymm[13]);
    vtype::storeu(arr + 112, ymm[14]);
    vtype::storeu(arr + 120, ymm[15]);
    vtype::mask_storeu(arr + 128, load_mask1, ymm[16]);
    vtype::mask_storeu(arr + 136, load_mask2, ymm[17]);
    vtype::mask_storeu(arr + 144, load_mask3, ymm[18]);
    vtype::mask_storeu(arr + 152, load_mask4, ymm[19]);
    vtype::mask_storeu(arr + 160, load_mask5, ymm[20]);
    vtype::mask_storeu(arr + 168, load_mask6, ymm[21]);
    vtype::mask_storeu(arr + 176, load_mask7, ymm[22]);
    vtype::mask_storeu(arr + 184, load_mask8, ymm[23]);
    if (N > 192) {
        vtype::mask_storeu(arr + 192, load_mask9, ymm[24]);
        vtype::mask_storeu(arr + 200, load_mask10, ymm[25]);
        vtype::mask_storeu(arr + 208, load_mask11, ymm[26]);
        vtype::mask_storeu(arr + 216, load_mask12, ymm[27]);
        vtype::mask_storeu(arr + 224, load_mask13, ymm[28]);
        vtype::mask_storeu(arr + 232, load_mask14, ymm[29]);
        vtype::mask_storeu(arr + 240, load_mask15, ymm[30]);
        vtype::mask_storeu(arr + 248, load_mask16, ymm[31]);
    }
}

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
        sort_256_64bit<vtype>(arr + left, (int32_t)(right + 1 - left));
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
        sort_128_64bit<vtype>(arr + left, (int32_t)(right + 1 - left));
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

template <>
void avx2_qselect<int32_t>(int32_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_64bit_<ymm_vector<int32_t>, int32_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qselect<uint32_t>(uint32_t *arr, int64_t k, int64_t arrsize, bool /*hasnan*/)
{
    if (arrsize > 1) {
        qselect_64bit_<ymm_vector<uint32_t>, uint32_t>(
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
        qselect_64bit_<ymm_vector<float>, float>(
            arr, k, 0, indx_last_elem, 2 * (int64_t)log2(indx_last_elem));
    }
}

template <>
void avx2_qsort<int32_t>(int32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_64bit_<ymm_vector<int32_t>, int32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<uint32_t>(uint32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_64bit_<ymm_vector<uint32_t>, uint32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx2_qsort<float>(float *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_64bit_<ymm_vector<float>, float>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // AVX2_QSORT_32BIT
