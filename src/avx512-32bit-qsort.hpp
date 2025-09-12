/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * Copyright (C) 2021 Serge Sans Paille
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 *          Serge Sans Paille <serge.guelton@telecom-bretagne.eu>
 * ****************************************************************/
#ifndef AVX512_QSORT_32BIT
#define AVX512_QSORT_32BIT

struct avx512_32bit_swizzle_ops;

template <>
struct zmm_vector<int32_t> {
    using type_t = int32_t;
    using reg_t = __m512i;
    using regi_t = __m512i;
    using halfreg_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;
#ifdef XSS_MINIMAL_NETWORK_SORT
    static constexpr int network_sort_threshold = numlanes;
#else
    static constexpr int network_sort_threshold = 512;
#endif
    static constexpr int partition_unroll_factor = 8;
    static constexpr simd_type vec_type = simd_type::AVX512;

    using swizzle_ops = avx512_32bit_swizzle_ops;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT32;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT32;
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_min()
    {
        return _mm512_set1_epi32(type_min());
    }

    static X86_SIMD_SORT_FORCE_INLINE opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NLT);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmpeq_epi32_mask(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t get_partial_loadmask(uint64_t num_to_read)
    {
        return ((0x1ull << num_to_read) - 0x1ull);
    }
    template <int scale>
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t merge(halfreg_t y1, halfreg_t y2)
    {
        reg_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_epi32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_epi32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_epi32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_epi32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE regi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8,
                       int v9,
                       int v10,
                       int v11,
                       int v12,
                       int v13,
                       int v14,
                       int v15,
                       int v16)
    {
        return _mm512_set_epi32(v1,
                                v2,
                                v3,
                                v4,
                                v5,
                                v6,
                                v7,
                                v8,
                                v9,
                                v10,
                                v11,
                                v12,
                                v13,
                                v14,
                                v15,
                                v16);
    }
    template <uint8_t mask>
    static X86_SIMD_SORT_FORCE_INLINE reg_t shuffle(reg_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static X86_SIMD_SORT_FORCE_INLINE void storeu(void *mem, reg_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static X86_SIMD_SORT_FORCE_INLINE halfreg_t max(halfreg_t x, halfreg_t y)
    {
        return _mm256_max_epi32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t min(halfreg_t x, halfreg_t y)
    {
        return _mm256_min_epi32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t reverse(reg_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_REVERSE_16LANES);
        return permutexvar(rev_index, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t sort_vec(reg_t x)
    {
        return sort_reg_16lanes<zmm_vector<type_t>>(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t cast_from(__m512i v)
    {
        return v;
    }
    static X86_SIMD_SORT_FORCE_INLINE __m512i cast_to(reg_t v)
    {
        return v;
    }
    static X86_SIMD_SORT_FORCE_INLINE bool all_false(opmask_t k)
    {
        return k == 0;
    }
    static X86_SIMD_SORT_FORCE_INLINE int double_compressstore(type_t *left_addr,
                                    type_t *right_addr,
                                    opmask_t k,
                                    reg_t reg)
    {
        return avx512_double_compressstore<zmm_vector<type_t>>(
                left_addr, right_addr, k, reg);
    }
};
template <>
struct zmm_vector<uint32_t> {
    using type_t = uint32_t;
    using reg_t = __m512i;
    using regi_t = __m512i;
    using halfreg_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;
#ifdef XSS_MINIMAL_NETWORK_SORT
    static constexpr int network_sort_threshold = numlanes;
#else
    static constexpr int network_sort_threshold = 512;
#endif
    static constexpr int partition_unroll_factor = 8;
    static constexpr simd_type vec_type = simd_type::AVX512;

    using swizzle_ops = avx512_32bit_swizzle_ops;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT32;
    }
    static type_t type_min()
    {
        return 0;
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_min()
    {
        return _mm512_set1_epi32(type_min());
    }

    template <int scale>
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t merge(halfreg_t y1, halfreg_t y2)
    {
        reg_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_epu32_mask(x, y, _MM_CMPINT_NLT);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmpeq_epu32_mask(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t get_partial_loadmask(uint64_t num_to_read)
    {
        return ((0x1ull << num_to_read) - 0x1ull);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_epu32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_epu32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_epu32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_epu32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE regi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8,
                       int v9,
                       int v10,
                       int v11,
                       int v12,
                       int v13,
                       int v14,
                       int v15,
                       int v16)
    {
        return _mm512_set_epi32(v1,
                                v2,
                                v3,
                                v4,
                                v5,
                                v6,
                                v7,
                                v8,
                                v9,
                                v10,
                                v11,
                                v12,
                                v13,
                                v14,
                                v15,
                                v16);
    }
    template <uint8_t mask>
    static X86_SIMD_SORT_FORCE_INLINE reg_t shuffle(reg_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static X86_SIMD_SORT_FORCE_INLINE void storeu(void *mem, reg_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static X86_SIMD_SORT_FORCE_INLINE halfreg_t max(halfreg_t x, halfreg_t y)
    {
        return _mm256_max_epu32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t min(halfreg_t x, halfreg_t y)
    {
        return _mm256_min_epu32(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t reverse(reg_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_REVERSE_16LANES);
        return permutexvar(rev_index, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t sort_vec(reg_t x)
    {
        return sort_reg_16lanes<zmm_vector<type_t>>(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t cast_from(__m512i v)
    {
        return v;
    }
    static X86_SIMD_SORT_FORCE_INLINE __m512i cast_to(reg_t v)
    {
        return v;
    }
    static X86_SIMD_SORT_FORCE_INLINE bool all_false(opmask_t k)
    {
        return k == 0;
    }
    static X86_SIMD_SORT_FORCE_INLINE int double_compressstore(type_t *left_addr,
                                    type_t *right_addr,
                                    opmask_t k,
                                    reg_t reg)
    {
        return avx512_double_compressstore<zmm_vector<type_t>>(
                left_addr, right_addr, k, reg);
    }
};
template <>
struct zmm_vector<float> {
    using type_t = float;
    using reg_t = __m512;
    using regi_t = __m512i;
    using halfreg_t = __m256;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;
#ifdef XSS_MINIMAL_NETWORK_SORT
    static constexpr int network_sort_threshold = numlanes;
#else
    static constexpr int network_sort_threshold = 512;
#endif
    static constexpr int partition_unroll_factor = 8;
    static constexpr simd_type vec_type = simd_type::AVX512;

    using swizzle_ops = avx512_32bit_swizzle_ops;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYF;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITYF;
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_max()
    {
        return _mm512_set1_ps(type_max());
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t zmm_min()
    {
        return _mm512_set1_ps(type_min());
    }

    static X86_SIMD_SORT_FORCE_INLINE opmask_t knot_opmask(opmask_t x)
    {
        return _mm512_knot(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmpeq_ps_mask(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE opmask_t get_partial_loadmask(uint64_t num_to_read)
    {
        return ((0x1ull << num_to_read) - 0x1ull);
    }
    static X86_SIMD_SORT_FORCE_INLINE int32_t convert_mask_to_int(opmask_t mask)
    {
        return mask;
    }
    template <int type>
    static X86_SIMD_SORT_FORCE_INLINE opmask_t fpclass(reg_t x)
    {
        return _mm512_fpclass_ps_mask(x, type);
    }
    template <int scale>
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_ps(index, base, scale);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t merge(halfreg_t y1, halfreg_t y2)
    {
        reg_t z1 = _mm512_castsi512_ps(
                _mm512_castsi256_si512(_mm256_castps_si256(y1)));
        return _mm512_insertf32x8(z1, y2, 1);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t loadu(void const *mem)
    {
        return _mm512_loadu_ps(mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_ps(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_ps(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm512_maskz_loadu_ps(mask, mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_ps(x, mask, mem);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_ps(x, mask, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_ps(mem, mask, x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_ps(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_ps(idx, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_ps(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_ps(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t set1(type_t v)
    {
        return _mm512_set1_ps(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE regi_t seti(int v1,
                       int v2,
                       int v3,
                       int v4,
                       int v5,
                       int v6,
                       int v7,
                       int v8,
                       int v9,
                       int v10,
                       int v11,
                       int v12,
                       int v13,
                       int v14,
                       int v15,
                       int v16)
    {
        return _mm512_set_epi32(v1,
                                v2,
                                v3,
                                v4,
                                v5,
                                v6,
                                v7,
                                v8,
                                v9,
                                v10,
                                v11,
                                v12,
                                v13,
                                v14,
                                v15,
                                v16);
    }
    template <uint8_t mask>
    static X86_SIMD_SORT_FORCE_INLINE reg_t shuffle(reg_t zmm)
    {
        return _mm512_shuffle_ps(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static X86_SIMD_SORT_FORCE_INLINE void storeu(void *mem, reg_t x)
    {
        return _mm512_storeu_ps(mem, x);
    }

    static X86_SIMD_SORT_FORCE_INLINE halfreg_t max(halfreg_t x, halfreg_t y)
    {
        return _mm256_max_ps(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE halfreg_t min(halfreg_t x, halfreg_t y)
    {
        return _mm256_min_ps(x, y);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t reverse(reg_t zmm)
    {
        const auto rev_index = _mm512_set_epi32(NETWORK_REVERSE_16LANES);
        return permutexvar(rev_index, zmm);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t sort_vec(reg_t x)
    {
        return sort_reg_16lanes<zmm_vector<type_t>>(x);
    }
    static X86_SIMD_SORT_FORCE_INLINE reg_t cast_from(__m512i v)
    {
        return _mm512_castsi512_ps(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE __m512i cast_to(reg_t v)
    {
        return _mm512_castps_si512(v);
    }
    static X86_SIMD_SORT_FORCE_INLINE bool all_false(opmask_t k)
    {
        return k == 0;
    }
    static X86_SIMD_SORT_FORCE_INLINE int double_compressstore(type_t *left_addr,
                                    type_t *right_addr,
                                    opmask_t k,
                                    reg_t reg)
    {
        return avx512_double_compressstore<zmm_vector<type_t>>(
                left_addr, right_addr, k, reg);
    }
};

struct avx512_32bit_swizzle_ops {
    template <typename vtype, int scale>
    static X86_SIMD_SORT_FORCE_INLINE typename vtype::reg_t
    swap_n(typename vtype::reg_t reg)
    {
        __m512i v = vtype::cast_to(reg);

        if constexpr (scale == 2) {
            v = _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0b10110001);
        }
        else if constexpr (scale == 4) {
            v = _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0b01001110);
        }
        else if constexpr (scale == 8) {
            v = _mm512_shuffle_i64x2(v, v, 0b10110001);
        }
        else if constexpr (scale == 16) {
            v = _mm512_shuffle_i64x2(v, v, 0b01001110);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v);
    }

    template <typename vtype, int scale>
    static X86_SIMD_SORT_FORCE_INLINE typename vtype::reg_t
    reverse_n(typename vtype::reg_t reg)
    {
        __m512i v = vtype::cast_to(reg);

        if constexpr (scale == 2) { return swap_n<vtype, 2>(reg); }
        else if constexpr (scale == 4) {
            __m512i mask = _mm512_set_epi32(
                    12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
            v = _mm512_permutexvar_epi32(mask, v);
        }
        else if constexpr (scale == 8) {
            __m512i mask = _mm512_set_epi32(
                    8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
            v = _mm512_permutexvar_epi32(mask, v);
        }
        else if constexpr (scale == 16) {
            return vtype::reverse(reg);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v);
    }

    template <typename vtype, int scale>
    static X86_SIMD_SORT_FORCE_INLINE typename vtype::reg_t
    merge_n(typename vtype::reg_t reg, typename vtype::reg_t other)
    {
        __m512i v1 = vtype::cast_to(reg);
        __m512i v2 = vtype::cast_to(other);

        if constexpr (scale == 2) {
            v1 = _mm512_mask_blend_epi32(0b0101010101010101, v1, v2);
        }
        else if constexpr (scale == 4) {
            v1 = _mm512_mask_blend_epi32(0b0011001100110011, v1, v2);
        }
        else if constexpr (scale == 8) {
            v1 = _mm512_mask_blend_epi32(0b0000111100001111, v1, v2);
        }
        else if constexpr (scale == 16) {
            v1 = _mm512_mask_blend_epi32(0b0000000011111111, v1, v2);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v1);
    }
};

#endif //AVX512_QSORT_32BIT
