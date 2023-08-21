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

template <typename vtype, typename reg_t>
X86_SIMD_SORT_INLINE reg_t sort_zmm_64bit(reg_t zmm);

template <typename vtype, typename reg_t>
X86_SIMD_SORT_INLINE reg_t bitonic_merge_zmm_64bit(reg_t zmm);

template <>
struct ymm_vector<float> {
    using type_t = float;
    using reg_t = __m256;
    using zmmi_t = __m256i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYF;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITYF;
    }
    static reg_t zmm_max()
    {
        return _mm256_set1_ps(type_max());
    }

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static opmask_t kxor_opmask(opmask_t x, opmask_t y)
    {
        return _kxor_mask8(x, y);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t le(reg_t x, reg_t y)
    {
        return _mm256_cmp_ps_mask(x, y, _CMP_LE_OQ);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm256_cmp_ps_mask(x, y, _CMP_GE_OQ);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm256_cmp_ps_mask(x, y, _CMP_EQ_OQ);
    }
    static opmask_t get_partial_loadmask(int size)
    {
        return (0x01 << size) - 0x01;
    }
    template <int type>
    static opmask_t fpclass(reg_t x)
    {
        return _mm256_fpclass_ps_mask(x, type);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_ps(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_ps(index, base, scale);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm256_loadu_ps((float *)mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm256_max_ps(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_compressstoreu_ps(mem, mask, x);
    }
    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskz_loadu_ps(mask, mem);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm256_mask_loadu_ps(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm256_mask_mov_ps(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_storeu_ps(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm256_min_ps(x, y);
    }
    static reg_t permutexvar(__m256i idx, reg_t zmm)
    {
        return _mm256_permutexvar_ps(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        __m128 v128 = _mm_max_ps(_mm256_castps256_ps128(v),
                                 _mm256_extractf32x4_ps(v, 1));
        __m128 v64 = _mm_max_ps(
                v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128 v32 = _mm_max_ps(
                v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return _mm_cvtss_f32(v32);
    }
    static type_t reducemin(reg_t v)
    {
        __m128 v128 = _mm_min_ps(_mm256_castps256_ps128(v),
                                 _mm256_extractf32x4_ps(v, 1));
        __m128 v64 = _mm_min_ps(
                v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128 v32 = _mm_min_ps(
                v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return _mm_cvtss_f32(v32);
    }
    static reg_t set1(type_t v)
    {
        return _mm256_set1_ps(v);
    }
    template <uint8_t mask, bool = (mask == 0b01010101)>
    static reg_t shuffle(reg_t zmm)
    {
        /* Hack!: have to make shuffles within 128-bit lanes work for both
         * 32-bit and 64-bit */
        return _mm256_shuffle_ps(zmm, zmm, 0b10110001);
        //if constexpr (mask == 0b01010101) {
        //}
        //else {
        //    /* Not used, so far */
        //    return _mm256_shuffle_ps(zmm, zmm, mask);
        //}
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm256_storeu_ps((float *)mem, x);
    }
};
template <>
struct ymm_vector<uint32_t> {
    using type_t = uint32_t;
    using reg_t = __m256i;
    using zmmi_t = __m256i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT32;
    }
    static type_t type_min()
    {
        return 0;
    }
    static reg_t zmm_max()
    {
        return _mm256_set1_epi32(type_max());
    }

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static opmask_t kxor_opmask(opmask_t x, opmask_t y)
    {
        return _kxor_mask8(x, y);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t le(reg_t x, reg_t y)
    {
        return _mm256_cmp_epu32_mask(x, y, _MM_CMPINT_LE);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm256_cmp_epu32_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm256_cmp_epu32_mask(x, y, _MM_CMPINT_EQ);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_epi32(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((__m256i *)mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm256_max_epu32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_compressstoreu_epi32(mem, mask, x);
    }
    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskz_loadu_epi32(mask, mem);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm256_mask_loadu_epi32(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm256_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_storeu_epi32(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm256_min_epu32(x, y);
    }
    static reg_t permutexvar(__m256i idx, reg_t zmm)
    {
        return _mm256_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        __m128i v128 = _mm_max_epu32(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1));
        __m128i v64 = _mm_max_epu32(
                v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128i v32 = _mm_max_epu32(
                v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return (type_t)_mm_cvtsi128_si32(v32);
    }
    static type_t reducemin(reg_t v)
    {
        __m128i v128 = _mm_min_epu32(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1));
        __m128i v64 = _mm_min_epu32(
                v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128i v32 = _mm_min_epu32(
                v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return (type_t)_mm_cvtsi128_si32(v32);
    }
    static reg_t set1(type_t v)
    {
        return _mm256_set1_epi32(v);
    }
    template <uint8_t mask, bool = (mask == 0b01010101)>
    static reg_t shuffle(reg_t zmm)
    {
        /* Hack!: have to make shuffles within 128-bit lanes work for both
         * 32-bit and 64-bit */
        return _mm256_shuffle_epi32(zmm, 0b10110001);
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm256_storeu_si256((__m256i *)mem, x);
    }
};
template <>
struct ymm_vector<int32_t> {
    using type_t = int32_t;
    using reg_t = __m256i;
    using zmmi_t = __m256i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT32;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT32;
    }
    static reg_t zmm_max()
    {
        return _mm256_set1_epi32(type_max());
    } // TODO: this should broadcast bits as is?

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static opmask_t kxor_opmask(opmask_t x, opmask_t y)
    {
        return _kxor_mask8(x, y);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t le(reg_t x, reg_t y)
    {
        return _mm256_cmp_epi32_mask(x, y, _MM_CMPINT_LE);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm256_cmp_epi32_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm256_cmp_epi32_mask(x, y, _MM_CMPINT_EQ);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_epi32(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((__m256i *)mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm256_max_epi32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_compressstoreu_epi32(mem, mask, x);
    }
    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm256_maskz_loadu_epi32(mask, mem);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm256_mask_loadu_epi32(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm256_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_mask_storeu_epi32(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm256_min_epi32(x, y);
    }
    static reg_t permutexvar(__m256i idx, reg_t zmm)
    {
        return _mm256_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        __m128i v128 = _mm_max_epi32(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1));
        __m128i v64 = _mm_max_epi32(
                v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128i v32 = _mm_max_epi32(
                v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return (type_t)_mm_cvtsi128_si32(v32);
    }
    static type_t reducemin(reg_t v)
    {
        __m128i v128 = _mm_min_epi32(_mm256_castsi256_si128(v),
                                     _mm256_extracti128_si256(v, 1));
        __m128i v64 = _mm_min_epi32(
                v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128i v32 = _mm_min_epi32(
                v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));
        return (type_t)_mm_cvtsi128_si32(v32);
    }
    static reg_t set1(type_t v)
    {
        return _mm256_set1_epi32(v);
    }
    template <uint8_t mask, bool = (mask == 0b01010101)>
    static reg_t shuffle(reg_t zmm)
    {
        /* Hack!: have to make shuffles within 128-bit lanes work for both
         * 32-bit and 64-bit */
        return _mm256_shuffle_epi32(zmm, 0b10110001);
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm256_storeu_si256((__m256i *)mem, x);
    }
};
template <>
struct zmm_vector<int64_t> {
    using type_t = int64_t;
    using reg_t = __m512i;
    using zmmi_t = __m512i;
    using halfreg_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;
    static constexpr int network_sort_threshold = 256;
    static constexpr int partition_unroll_factor = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT64;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT64;
    }
    static reg_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    } // TODO: this should broadcast bits as is?

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static opmask_t kxor_opmask(opmask_t x, opmask_t y)
    {
        return _kxor_mask8(x, y);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t le(reg_t x, reg_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_LE);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_EQ);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_epi64(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_epi64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm512_maskz_loadu_epi64(mask, mem);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_epi64(x, y);
    }
    static reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_epi64(v);
    }
    static type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_epi64(v);
    }
    static reg_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static reg_t shuffle(reg_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm512_storeu_si512(mem, x);
    }
    static reg_t reverse(reg_t zmm)
    {
        const zmmi_t rev_index = seti(NETWORK_64BIT_2);
        return permutexvar(rev_index, zmm);
    }
    static reg_t bitonic_merge(reg_t x)
    {
        return bitonic_merge_zmm_64bit<zmm_vector<type_t>>(x);
    }
    static reg_t sort_vec(reg_t x)
    {
        return sort_zmm_64bit<zmm_vector<type_t>>(x);
    }
};
template <>
struct zmm_vector<uint64_t> {
    using type_t = uint64_t;
    using reg_t = __m512i;
    using zmmi_t = __m512i;
    using halfreg_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;
    static constexpr int network_sort_threshold = 256;
    static constexpr int partition_unroll_factor = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT64;
    }
    static type_t type_min()
    {
        return 0;
    }
    static reg_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    }

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_epi64(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_EQ);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_epu64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_epu64(x, y);
    }
    static reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_epu64(v);
    }
    static type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_epu64(v);
    }
    static reg_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static reg_t shuffle(reg_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm512_storeu_si512(mem, x);
    }
    static reg_t reverse(reg_t zmm)
    {
        const zmmi_t rev_index = seti(NETWORK_64BIT_2);
        return permutexvar(rev_index, zmm);
    }
    static reg_t bitonic_merge(reg_t x)
    {
        return bitonic_merge_zmm_64bit<zmm_vector<type_t>>(x);
    }
    static reg_t sort_vec(reg_t x)
    {
        return sort_zmm_64bit<zmm_vector<type_t>>(x);
    }
};
template <>
struct zmm_vector<double> {
    using type_t = double;
    using reg_t = __m512d;
    using zmmi_t = __m512i;
    using halfreg_t = __m512d;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;
    static constexpr int network_sort_threshold = 256;
    static constexpr int partition_unroll_factor = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITY;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITY;
    }
    static reg_t zmm_max()
    {
        return _mm512_set1_pd(type_max());
    }

    static zmmi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm512_maskz_loadu_pd(mask, mem);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_GE_OQ);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
    }
    static opmask_t get_partial_loadmask(int size)
    {
        return (0x01 << size) - 0x01;
    }
    template <int type>
    static opmask_t fpclass(reg_t x)
    {
        return _mm512_fpclass_pd_mask(x, type);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_pd(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_pd(index, base, scale);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm512_loadu_pd(mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_pd(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_pd(mem, mask, x);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_pd(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_pd(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_pd(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_pd(x, y);
    }
    static reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_pd(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_pd(v);
    }
    static type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_pd(v);
    }
    static reg_t set1(type_t v)
    {
        return _mm512_set1_pd(v);
    }
    template <uint8_t mask>
    static reg_t shuffle(reg_t zmm)
    {
        return _mm512_shuffle_pd(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm512_storeu_pd(mem, x);
    }
    static reg_t reverse(reg_t zmm)
    {
        const zmmi_t rev_index = seti(NETWORK_64BIT_2);
        return permutexvar(rev_index, zmm);
    }
    static reg_t bitonic_merge(reg_t x)
    {
        return bitonic_merge_zmm_64bit<zmm_vector<type_t>>(x);
    }
    static reg_t sort_vec(reg_t x)
    {
        return sort_zmm_64bit<zmm_vector<type_t>>(x);
    }
};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t sort_zmm_64bit(reg_t zmm)
{
    const typename vtype::zmmi_t rev_index = vtype::seti(NETWORK_64BIT_2);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::seti(NETWORK_64BIT_1), zmm), 0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(zmm, vtype::permutexvar(rev_index, zmm), 0xF0);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::seti(NETWORK_64BIT_3), zmm), 0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t bitonic_merge_zmm_64bit(reg_t zmm)
{

    // 1) half_cleaner[8]: compare 0-4, 1-5, 2-6, 3-7
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_4), zmm),
            0xF0);
    // 2) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_3), zmm),
            0xCC);
    // 3) half_cleaner[1]
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
    using reg_t = typename vtype::reg_t;
    __m512i rand_index = _mm512_set_epi64(left + size,
                                          left + 2 * size,
                                          left + 3 * size,
                                          left + 4 * size,
                                          left + 5 * size,
                                          left + 6 * size,
                                          left + 7 * size,
                                          left + 8 * size);
    reg_t rand_vec = vtype::template i64gather<sizeof(type_t)>(rand_index, arr);
    // pivot will never be a nan, since there are no nan's!
    reg_t sort = sort_zmm_64bit<vtype>(rand_vec);
    return ((type_t *)&sort)[4];
}

#endif
