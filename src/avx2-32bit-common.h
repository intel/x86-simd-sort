/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX2_32BIT_COMMON
#define AVX2_32BIT_COMMON
#include "avx2-common-qsort.h"
#include "avx2-emu-funcs.hpp"

/*
 * Constants used in sorting 8 elements in a ymm registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ymm                  7, 6, 5, 4, 3, 2, 1, 0
#define NETWORK_32BIT_1 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK_32BIT_2 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_32BIT_3 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK_32BIT_4 3, 2, 1, 0, 7, 6, 5, 4

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

/*
 * Assumes ymm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t sort_ymm_32bit(ymm_t ymm)
{
    const typename vtype::opmask_t oxAA = _mm256_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
    const typename vtype::opmask_t oxCC = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
    const typename vtype::opmask_t oxF0 = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0);
    
    const typename vtype::ymmi_t rev_index = vtype::seti(NETWORK_32BIT_2);
    ymm = cmp_merge<vtype>(
            ymm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(vtype::seti(NETWORK_32BIT_1), ymm),
            oxCC);
    ymm = cmp_merge<vtype>(
            ymm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    ymm = cmp_merge<vtype>(ymm, vtype::permutexvar(rev_index, ymm), oxF0);
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::permutexvar(vtype::seti(NETWORK_32BIT_3), ymm),
            oxCC);
    ymm = cmp_merge<vtype>(
            ymm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    return ymm;
}

template <>
struct ymm_vector<int32_t> {
    using type_t = int32_t;
    using ymm_t = __m256i;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper32;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT32;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT32;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_epi32(type_max());
    } // TODO: this should broadcast bits as is?

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static opmask_t kxor_opmask(opmask_t x, opmask_t y)
    {
        return _mm256_xor_si256(x, y);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return ~x;
    }
    static opmask_t le(ymm_t x, ymm_t y)
    {
        return ~_mm256_cmpgt_epi32(x, y);
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        opmask_t equal = eq(x,y);
        opmask_t greater = _mm256_cmpgt_epi32(x, y);
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(equal), _mm256_castsi256_ps(greater)));
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_cmpeq_epi32(x, y);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i32gather_epi32(src, base, index, mask, scale);
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i32gather_epi32((int const *) base, index, scale);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((ymm_t const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epi32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore32<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskload_epi32((const int *) mem, mask);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_epi32((type_t *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), _mm256_castsi256_ps(mask)));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_epi32((type_t *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epi32(x, y);
    }
    static ymm_t permutexvar(__m256i idx, ymm_t ymm)
    {
        return _mm256_permutevar8x32_epi32(ymm, idx);
        //return avx2_emu_permutexvar_epi32(idx, ymm);
    }
    static ymm_t permutevar(ymm_t ymm, __m256i idx)
    {
        return _mm256_permutevar8x32_epi32 (ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const __m256i rev_index = _mm256_set_epi32(NETWORK_32BIT_2);
        return permutexvar(rev_index, ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        return _mm256_extract_epi32(v, index);
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max32<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min32<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_epi32(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_shuffle_epi32(ymm, mask);
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_si256((__m256i *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_32bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_32bit<ymm_vector<type_t>>(x);
    }
};
template <>
struct ymm_vector<uint32_t> {
    using type_t = uint32_t;
    using ymm_t = __m256i;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper32;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT32;
    }
    static type_t type_min()
    {
        return 0;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_epi32(type_max());
    }

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i32gather_epi32(src, base, index, mask, scale);
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i32gather_epi32((int const *) base, index, scale);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return ~x;
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        ymm_t maxi = max(x,y);
        return eq(maxi, x);
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_cmpeq_epi32(x, y);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((ymm_t const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epu32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore32<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_epi32((const int *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), _mm256_castsi256_ps(mask)));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_epi32((int *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epu32(x, y);
    }
    static ymm_t permutexvar(__m256i idx, ymm_t ymm)
    {
        return _mm256_permutevar8x32_epi32(ymm, idx);
    }
    static ymm_t permutevar(ymm_t ymm, __m256i idx)
    {
        return _mm256_permutevar8x32_epi32 (ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const __m256i rev_index = _mm256_set_epi32(NETWORK_32BIT_2);
        return permutexvar(rev_index, ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        return _mm256_extract_epi32(v, index);
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max32<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min32<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_epi32(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_shuffle_epi32(ymm, mask);
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_si256((__m256i *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_32bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_32bit<ymm_vector<type_t>>(x);
    }
};
template <>
struct ymm_vector<float> {
    using type_t = float;
    using ymm_t = __m256;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper32;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYF;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITYF;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_ps(type_max());
    }

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static ymm_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskload_ps((const float *) mem, mask);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return ~x;
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_GE_OQ));
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_castps_si256(_mm256_cmp_ps(x, y, _CMP_EQ_OQ));
    }
    template <int type>
    static opmask_t fpclass(ymm_t x)
    {
        return avx2_emu_fpclassify32<type_t>(x, type);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i32gather_ps(src, base, index, _mm256_castsi256_ps(mask), scale);;
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i32gather_ps((float *) base, index, scale);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_ps((float const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_ps(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore32<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_ps((type_t *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_blendv_ps(x, y, _mm256_castsi256_ps(mask));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_ps((type_t *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_ps(x, y);
    }
    static ymm_t permutexvar(__m256i idx, ymm_t ymm)
    {
        return _mm256_permutevar8x32_ps(ymm, idx);
    }
    static ymm_t permutevar(ymm_t ymm, __m256i idx)
    {
        return _mm256_permutevar8x32_ps(ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const __m256i rev_index = _mm256_set_epi32(NETWORK_32BIT_2);
        return permutexvar(rev_index, ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        int32_t x = _mm256_extract_epi32(_mm256_castps_si256(v), index);
        float y;
        std::memcpy(&y, &x, sizeof(y));
        return y;
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max32<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min32<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_ps(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(ymm), mask));
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_ps((float *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_32bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_32bit<ymm_vector<type_t>>(x);
    }
};
X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(float *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask8 loadmask = 0xFF;
    while (arrsize > 0) {
        if (arrsize < 8) { loadmask = (0x01 << arrsize) - 0x01; }
        __m256 in_ymm = ymm_vector<float>::maskz_loadu(loadmask, arr);
        __m256i nanmask = _mm256_castps_si256(_mm256_cmp_ps(in_ymm, in_ymm, _CMP_NEQ_UQ));
        nan_count += _popcnt32(avx2_mask_helper32(nanmask));
        ymm_vector<float>::mask_storeu(arr, nanmask, YMM_MAX_FLOAT);
        arr += 8;
        arrsize -= 8;
    }
    return nan_count;
}

X86_SIMD_SORT_INLINE void
replace_inf_with_nan(float *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = std::nan("1");
        nan_count -= 1;
    }
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_32bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 8
    int64_t size = (right - left) / 8;
    using ymm_t = typename vtype::ymm_t;
    __m256i rand_index = _mm256_set_epi32(left + size,
                                          left + 2 * size,
                                          left + 3 * size,
                                          left + 4 * size,
                                          left + 5 * size,
                                          left + 6 * size,
                                          left + 7 * size,
                                          left + 8 * size);
    ymm_t rand_vec = vtype::template i64gather<sizeof(type_t)>(rand_index, arr);
    // pivot will never be a nan, since there are no nan's!
    ymm_t sort = sort_ymm_32bit<vtype>(rand_vec);
    return ((type_t *)&sort)[4];
}

#endif
