/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX2_64BIT_COMMON
#define AVX2_64BIT_COMMON
#include "avx2-common-qsort.h"
#include "avx2-emu-funcs.hpp"

/*
 * Constants used in sorting 8 elements in a ymm registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ymm                  3, 2, 1, 0
#define NETWORK_64BIT_R 0, 1, 2, 3
#define NETWORK_64BIT_1 1, 0, 3, 2

namespace x86_simd_sort{
namespace avx2{

// Assumes ymm is bitonic and performs a recursive half cleaner
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t bitonic_merge_ymm_64bit(ymm_t ymm)
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

/*
 * Assumes ymm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename ymm_t = typename vtype::ymm_t>
X86_SIMD_SORT_INLINE ymm_t sort_ymm_64bit(ymm_t ymm)
{
    const typename vtype::opmask_t oxAA = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0);
    const typename vtype::opmask_t oxCC = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0);
    ymm = cmp_merge<vtype>(
            ymm, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    ymm = cmp_merge<vtype>(
            ymm,
            vtype::template permutexvar<SHUFFLE_MASK(0,1,2,3)>(ymm),
            oxCC);
    ymm = cmp_merge<vtype>(
            ymm, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(ymm), oxAA);
    return ymm;
}

template <>
struct ymm_vector<int64_t> {
    using type_t = int64_t;
    using ymm_t = __m256i;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper64;
    static const uint8_t numlanes = 4;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT64;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT64;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_epi64x(type_max());
    } // TODO: this should broadcast bits as is?

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4)
    {
        return _mm256_set_epi64x(v1, v2, v3, v4);
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
        return ~_mm256_cmpgt_epi64(x, y);
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        opmask_t equal = eq(x,y);
        opmask_t greater = _mm256_cmpgt_epi64(x, y);
        return _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(equal), _mm256_castsi256_pd(greater)));
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_cmpeq_epi64(x, y);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i64gather_epi64(src, base, index, mask, scale);
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i64gather_epi64((long long int const *) base, index, scale);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((ymm_t const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return avx2_emu_max<type_t>(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore64<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskload_epi64((const long long int *) mem, mask);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_epi64((long long int *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y), _mm256_castsi256_pd(mask)));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_epi64((long long int *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return avx2_emu_min<type_t>(x, y);
    }
    template <int32_t idx>
    static ymm_t permutexvar(ymm_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    template <int32_t idx>
    static ymm_t permutevar(ymm_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const int32_t rev_index = SHUFFLE_MASK(0,1,2,3);
        return permutexvar<rev_index>(ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        return _mm256_extract_epi64(v, index);
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max64<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min64<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_epi64x(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(ymm), mask));
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_si256((__m256i *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_64bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_64bit<ymm_vector<type_t>>(x);
    }
};
template <>
struct ymm_vector<uint64_t> {
    using type_t = uint64_t;
    using ymm_t = __m256i;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper64;
    static const uint8_t numlanes = 4;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT64;
    }
    static type_t type_min()
    {
        return 0;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_epi64x(type_max());
    }

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4)
    {
        return _mm256_set_epi64x(v1, v2, v3, v4);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i64gather_epi64(src, base, index, mask, scale);
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i64gather_epi64((long long int const *) base, index, scale);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return ~x;
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        // TODO real implementation
        uint64_t _x[4];
        uint64_t _y[4];
        storeu(_x, x);
        storeu(_y, y);
        uint64_t res[4];
        for (int i = 0; i < 4; i++){res[i] = _x[i] >= _y[i] ? 0xFFFFFFFFFFFFFFFF : 0;}
        return loadu(res);
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_cmpeq_epi64(x, y);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((ymm_t const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return avx2_emu_max<type_t>(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore64<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_epi64((const long long int *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y), _mm256_castsi256_pd(mask)));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_epi64((long long int *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return avx2_emu_min<type_t>(x, y);
    }
    template <int32_t idx>
    static ymm_t permutexvar(ymm_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    template <int32_t idx>
    static ymm_t permutevar(ymm_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const int32_t rev_index = SHUFFLE_MASK(0,1,2,3);
        return permutexvar<rev_index>(ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        return _mm256_extract_epi64(v, index);
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max64<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min64<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_epi64x(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(ymm), mask));
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_si256((__m256i *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_64bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_64bit<ymm_vector<type_t>>(x);
    }
};
template <>
struct ymm_vector<double> {
    using type_t = double;
    using ymm_t = __m256d;
    using ymmi_t = __m256i;
    using opmask_t = avx2_mask_helper64;
    static const uint8_t numlanes = 4;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITY;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITY;
    }
    static ymm_t ymm_max()
    {
        return _mm256_set1_pd(type_max());
    }

    static ymmi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4)
    {
        return _mm256_set_epi64x(v1, v2, v3, v4);
    }

    static ymm_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskload_pd((const double *) mem, mask);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return ~x;
    }
    static opmask_t ge(ymm_t x, ymm_t y)
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(x, y, _CMP_GE_OQ));
    }
    static opmask_t eq(ymm_t x, ymm_t y)
    {
        return _mm256_castpd_si256(_mm256_cmp_pd(x, y, _CMP_EQ_OQ));
    }
    template <int type>
    static opmask_t fpclass(ymm_t x)
    {
        return avx2_emu_fpclassify64<type_t>(x, type);
    }
    template <int scale>
    static ymm_t
    mask_i64gather(ymm_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i64gather_pd(src, base, index, _mm256_castsi256_pd(mask), scale);;
    }
    template <int scale>
    static ymm_t i64gather(__m256i index, void const *base)
    {
        return _mm256_i64gather_pd((double *) base, index, scale);
    }
    static ymm_t loadu(void const *mem)
    {
        return _mm256_loadu_pd((double const *) mem);
    }
    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_pd(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, ymm_t x)
    {
        return avx2_emu_mask_compressstoreu<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void * left_addr, void * right_addr, opmask_t k, ymm_t reg){
        return avx2_double_compressstore64<type_t>(left_addr, right_addr, k, reg);
    }
    static ymm_t mask_loadu(ymm_t x, opmask_t mask, void const *mem)
    {
        ymm_t dst = _mm256_maskload_pd((type_t *) mem, mask);
        return mask_mov(x, mask, dst);
    }
    static ymm_t mask_mov(ymm_t x, opmask_t mask, ymm_t y)
    {
        return _mm256_blendv_pd(x, y, _mm256_castsi256_pd(mask));
    }
    static void mask_storeu(void *mem, opmask_t mask, ymm_t x)
    {
        return _mm256_maskstore_pd((type_t *) mem, mask, x);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_pd(x, y);
    }
    template <int32_t idx>
    static ymm_t permutexvar(ymm_t ymm)
    {
        return _mm256_permute4x64_pd(ymm, idx);
    }
    template <int32_t idx>
    static ymm_t permutevar(ymm_t ymm)
    {
        return _mm256_permute4x64_pd(ymm, idx);
    }
    static ymm_t reverse(ymm_t ymm){
        const int32_t rev_index = SHUFFLE_MASK(0,1,2,3);
        return permutexvar<rev_index>(ymm);
    }
    template <int index>
    static type_t extract(ymm_t v){
        int64_t x = _mm256_extract_epi64(_mm256_castpd_si256(v), index);
        double y;
        std::memcpy(&y, &x, sizeof(y));
        return y;
    }
    static type_t reducemax(ymm_t v)
    {
        return avx2_emu_reduce_max64<type_t>(v);
    }
    static type_t reducemin(ymm_t v)
    {
        return avx2_emu_reduce_min64<type_t>(v);
    }
    static ymm_t set1(type_t v)
    {
        return _mm256_set1_pd(v);
    }
    template <uint8_t mask>
    static ymm_t shuffle(ymm_t ymm)
    {
        return _mm256_permute_pd(ymm, mask);
    }
    static void storeu(void *mem, ymm_t x)
    {
        _mm256_storeu_pd((double *) mem, x);
    }
    static ymm_t bitonic_merge(ymm_t x){
        return bitonic_merge_ymm_64bit<ymm_vector<type_t>>(x);
    }
    static ymm_t sort_vec(ymm_t x){
        return sort_ymm_64bit<ymm_vector<type_t>>(x);
    }
};
X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(double *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask8 loadmask = 0xF;
    while (arrsize > 0) {
        if (arrsize < 4) { loadmask = ((0x01 << arrsize) - 0x01) & 0xF; }
        __m256d in_ymm = ymm_vector<double>::maskz_loadu(loadmask, arr);
        __m256i nanmask = _mm256_castpd_si256(_mm256_cmp_pd(in_ymm, in_ymm, _CMP_NEQ_UQ));
        nan_count += _popcnt32(avx2_mask_helper64(nanmask));
        ymm_vector<double>::mask_storeu(arr, nanmask, YMM_MAX_DOUBLE);
        arr += 4;
        arrsize -= 4;
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

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_64bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 8
    int64_t size = (right - left) / 4;
    using ymm_t = typename vtype::ymm_t;
    __m256i rand_index = _mm256_set_epi64x(left + size,
                                          left + 2 * size,
                                          left + 3 * size,
                                          left + 4 * size);
    ymm_t rand_vec = vtype::template i64gather<sizeof(type_t)>(rand_index, arr);
    // pivot will never be a nan, since there are no nan's!
    ymm_t sort = sort_ymm_64bit<vtype>(rand_vec);
    return ((type_t *)&sort)[2];
}
}
}
#endif
