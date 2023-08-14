/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * Copyright (C) 2021 Serge Sans Paille
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 *          Serge Sans Paille <serge.guelton@telecom-bretagne.eu>
 * ****************************************************************/
#ifndef AVX512_QSORT_32BIT
#define AVX512_QSORT_32BIT

#include "avx512-common-qsort.h"
#include "xss-network-qsort.hpp"

/*
 * Constants used in sorting 16 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
#define NETWORK_32BIT_1 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
#define NETWORK_32BIT_2 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK_32BIT_3 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_32BIT_4 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK_32BIT_5 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define NETWORK_32BIT_6 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4
#define NETWORK_32BIT_7 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8

template <typename vtype, typename zmm_t>
X86_SIMD_SORT_INLINE zmm_t sort_zmm_32bit(zmm_t zmm);

template <typename vtype, typename zmm_t>
X86_SIMD_SORT_INLINE zmm_t bitonic_merge_zmm_32bit(zmm_t zmm);

template <>
struct zmm_vector<int32_t> {
    using type_t = int32_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT32;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT32;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NLT);
    }
    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi32(x, y);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi32(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epi32(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epi32(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epi32(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epi32(x, y);
    }
    static zmm_t reverse(zmm_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_32BIT_5);
        return permutexvar(rev_index, zmm);
    }
    static zmm_t bitonic_merge(zmm_t x)
    {
        return bitonic_merge_zmm_32bit<zmm_vector<type_t>>(x);
    }
    static zmm_t sort_vec(zmm_t x)
    {
        return sort_zmm_32bit<zmm_vector<type_t>>(x);
    }
};
template <>
struct zmm_vector<uint32_t> {
    using type_t = uint32_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT32;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    } // TODO: this should broadcast bits as is?

    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu32_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu32(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epu32(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epu32(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epu32(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epu32(x, y);
    }
    static zmm_t reverse(zmm_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_32BIT_5);
        return permutexvar(rev_index, zmm);
    }
    static zmm_t bitonic_merge(zmm_t x)
    {
        return bitonic_merge_zmm_32bit<zmm_vector<type_t>>(x);
    }
    static zmm_t sort_vec(zmm_t x)
    {
        return sort_zmm_32bit<zmm_vector<type_t>>(x);
    }
};
template <>
struct zmm_vector<float> {
    using type_t = float;
    using zmm_t = __m512;
    using ymm_t = __m256;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYF;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITYF;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_ps(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
    }
    static opmask_t get_partial_loadmask(int size)
    {
        return (0x0001 << size) - 0x0001;
    }
    template <int type>
    static opmask_t fpclass(zmm_t x)
    {
        return _mm512_fpclass_ps_mask(x, type);
    }
    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_ps(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi512_ps(
                _mm512_castsi256_si512(_mm256_castps_si256(y1)));
        return _mm512_insertf32x8(z1, y2, 1);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_ps(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_ps(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_ps(mem, mask, x);
    }
    static zmm_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm512_maskz_loadu_ps(mask, mem);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_ps(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_ps(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_ps(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_ps(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_ps(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_ps(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_ps(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_ps(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_ps(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_ps(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_ps(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_ps(x, y);
    }
    static zmm_t reverse(zmm_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_32BIT_5);
        return permutexvar(rev_index, zmm);
    }
    static zmm_t bitonic_merge(zmm_t x)
    {
        return bitonic_merge_zmm_32bit<zmm_vector<type_t>>(x);
    }
    static zmm_t sort_vec(zmm_t x)
    {
        return sort_zmm_32bit<zmm_vector<type_t>>(x);
    }
};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t sort_zmm_32bit(zmm_t zmm)
{
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_3), zmm),
            0xF0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm),
            0xFF00);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_6), zmm),
            0xF0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t bitonic_merge_zmm_32bit(zmm_t zmm)
{
    // 1) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_7), zmm),
            0xFF00);
    // 2) half_cleaner[8]: compare 1-5, 2-6, 3-7 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_6), zmm),
            0xF0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    // 3) half_cleaner[1]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    return zmm;
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_32bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 16
    int64_t size = (right - left) / 16;
    using zmm_t = typename vtype::zmm_t;
    using ymm_t = typename vtype::ymm_t;
    __m512i rand_index1 = _mm512_set_epi64(left + size,
                                           left + 2 * size,
                                           left + 3 * size,
                                           left + 4 * size,
                                           left + 5 * size,
                                           left + 6 * size,
                                           left + 7 * size,
                                           left + 8 * size);
    __m512i rand_index2 = _mm512_set_epi64(left + 9 * size,
                                           left + 10 * size,
                                           left + 11 * size,
                                           left + 12 * size,
                                           left + 13 * size,
                                           left + 14 * size,
                                           left + 15 * size,
                                           left + 16 * size);
    ymm_t rand_vec1
            = vtype::template i64gather<sizeof(type_t)>(rand_index1, arr);
    ymm_t rand_vec2
            = vtype::template i64gather<sizeof(type_t)>(rand_index2, arr);
    zmm_t rand_vec = vtype::merge(rand_vec1, rand_vec2);
    zmm_t sort = sort_zmm_32bit<vtype>(rand_vec);
    // pivot will never be a nan, since there are no nan's!
    return ((type_t *)&sort)[8];
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
     * Base case: use bitonic networks to sort arrays <= 256
     */
    if (right + 1 - left <= 256) {
        xss::sort_n<vtype, 256>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_32bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 2>(
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
     * Base case: use bitonic networks to sort arrays <= 256
     */
    if (right + 1 - left <= 256) {
        xss::sort_n<vtype, 256>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_32bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512_unrolled<vtype, 2>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if ((pivot != smallest) && (pos < pivot_index))
        qselect_32bit_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_32bit_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

/* Specialized template function for 32-bit qselect_ funcs*/
template <>
void qselect_<zmm_vector<int32_t>>(
        int32_t *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_32bit_<zmm_vector<int32_t>>(arr, k, left, right, maxiters);
}

template <>
void qselect_<zmm_vector<uint32_t>>(
        uint32_t *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_32bit_<zmm_vector<uint32_t>>(arr, k, left, right, maxiters);
}

template <>
void qselect_<zmm_vector<float>>(
        float *arr, int64_t k, int64_t left, int64_t right, int64_t maxiters)
{
    qselect_32bit_<zmm_vector<float>>(arr, k, left, right, maxiters);
}

/* Specialized template function for 32-bit qsort_ funcs*/
template <>
void qsort_<zmm_vector<int32_t>>(int32_t *arr,
                                 int64_t left,
                                 int64_t right,
                                 int64_t maxiters)
{
    qsort_32bit_<zmm_vector<int32_t>>(arr, left, right, maxiters);
}

template <>
void qsort_<zmm_vector<uint32_t>>(uint32_t *arr,
                                  int64_t left,
                                  int64_t right,
                                  int64_t maxiters)
{
    qsort_32bit_<zmm_vector<uint32_t>>(arr, left, right, maxiters);
}

template <>
void qsort_<zmm_vector<float>>(float *arr,
                               int64_t left,
                               int64_t right,
                               int64_t maxiters)
{
    qsort_32bit_<zmm_vector<float>>(arr, left, right, maxiters);
}
#endif //AVX512_QSORT_32BIT
