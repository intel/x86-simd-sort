#ifndef AVX512_ZMM_CLASSES
#define AVX512_ZMM_CLASSES

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>

#ifdef _MSC_VER
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static __forceinline
#elif defined(__CYGWIN__)
/*
 * Force inline in cygwin to work around a compiler bug. See
 * https://github.com/numpy/numpy/pull/22315#issuecomment-1267757584
 */
#define X86_SIMD_SORT_INLINE static __attribute__((always_inline))
#define X86_SIMD_SORT_FINLINE static __attribute__((always_inline))
#elif defined(__GNUC__)
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static __attribute__((always_inline))
#else
#define X86_SIMD_SORT_INLINE static
#define X86_SIMD_SORT_FINLINE static
#endif

#define X86_SIMD_SORT_INFINITY std::numeric_limits<double>::infinity()
#define X86_SIMD_SORT_INFINITYF std::numeric_limits<float>::infinity()
#define X86_SIMD_SORT_INFINITYH 0x7c00
#define X86_SIMD_SORT_NEGINFINITYH 0xfc00
#define X86_SIMD_SORT_MAX_UINT16 std::numeric_limits<uint16_t>::max()
#define X86_SIMD_SORT_MAX_INT16 std::numeric_limits<int16_t>::max()
#define X86_SIMD_SORT_MIN_INT16 std::numeric_limits<int16_t>::min()
#define X86_SIMD_SORT_MAX_UINT32 std::numeric_limits<uint32_t>::max()
#define X86_SIMD_SORT_MAX_INT32 std::numeric_limits<int32_t>::max()
#define X86_SIMD_SORT_MIN_INT32 std::numeric_limits<int32_t>::min()
#define X86_SIMD_SORT_MAX_UINT64 std::numeric_limits<uint64_t>::max()
#define X86_SIMD_SORT_MAX_INT64 std::numeric_limits<int64_t>::max()
#define X86_SIMD_SORT_MIN_INT64 std::numeric_limits<int64_t>::min()
#define ZMM_MAX_DOUBLE _mm512_set1_pd(X86_SIMD_SORT_INFINITY)
#define ZMM_MAX_UINT64 _mm512_set1_epi64(X86_SIMD_SORT_MAX_UINT64)
#define ZMM_MAX_INT64 _mm512_set1_epi64(X86_SIMD_SORT_MAX_INT64)
#define ZMM_MAX_FLOAT _mm512_set1_ps(X86_SIMD_SORT_INFINITYF)
#define ZMM_MAX_UINT _mm512_set1_epi32(X86_SIMD_SORT_MAX_UINT32)
#define ZMM_MAX_INT _mm512_set1_epi32(X86_SIMD_SORT_MAX_INT32)
#define ZMM_MAX_HALF _mm512_set1_epi16(X86_SIMD_SORT_INFINITYH)
#define YMM_MAX_HALF _mm256_set1_epi16(X86_SIMD_SORT_INFINITYH)
#define ZMM_MAX_UINT16 _mm512_set1_epi16(X86_SIMD_SORT_MAX_UINT16)
#define ZMM_MAX_INT16 _mm512_set1_epi16(X86_SIMD_SORT_MAX_INT16)
#define SHUFFLE_MASK(a, b, c, d) (a << 6) | (b << 4) | (c << 2) | d

// ZMM register: 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
static const uint16_t network[6][32]
        = {{7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13, 12, 11, 10, 9,  8,
            23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24},
           {15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16},
           {4,  5,  6,  7,  0,  1,  2,  3,  12, 13, 14, 15, 8,  9,  10, 11,
            20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27},
           {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
           {8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,
            24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23},
           {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15}};

template <typename type>
struct zmm_vector;

typedef union {
    _Float16 f_;
    uint16_t i_;
} Fp16Bits;

template <>
struct zmm_vector<_Float16> {
    using type_t = _Float16;
    using zmm_t = __m512h;
    using ymm_t = __m256h;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static __m512i get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        Fp16Bits val;
        val.i_ = X86_SIMD_SORT_INFINITYH;
        return val.f_;
    }
    static type_t type_min()
    {
        Fp16Bits val;
        val.i_ = X86_SIMD_SORT_NEGINFINITYH;
        return val.f_;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_ph(type_max());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }

    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_ph_mask(x, y, _CMP_GE_OQ);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_ph(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_ph(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        __m512i temp = _mm512_castph_si512(x);
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, temp);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_castsi512_ph(
                _mm512_mask_loadu_epi16(_mm512_castph_si512(x), mask, mem));
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_castsi512_ph(_mm512_mask_mov_epi16(
                _mm512_castph_si512(x), mask, _mm512_castph_si512(y)));
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, _mm512_castph_si512(x));
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_ph(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_ph(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_ph(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_ph(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_ph(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        __m512i temp = _mm512_shufflehi_epi16(_mm512_castph_si512(zmm),
                                              (_MM_PERM_ENUM)mask);
        return _mm512_castsi512_ph(
                _mm512_shufflelo_epi16(temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_ph(mem, x);
    }
};

struct float16 {
    uint16_t val;
};

template <>
struct zmm_vector<float16> {
    using type_t = uint16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYH;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_NEGINFINITYH;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }

    static opmask_t ge(zmm_t x, zmm_t y)
    {
        zmm_t sign_x = _mm512_and_si512(x, _mm512_set1_epi16(0x8000));
        zmm_t sign_y = _mm512_and_si512(y, _mm512_set1_epi16(0x8000));
        zmm_t exp_x = _mm512_and_si512(x, _mm512_set1_epi16(0x7c00));
        zmm_t exp_y = _mm512_and_si512(y, _mm512_set1_epi16(0x7c00));
        zmm_t mant_x = _mm512_and_si512(x, _mm512_set1_epi16(0x3ff));
        zmm_t mant_y = _mm512_and_si512(y, _mm512_set1_epi16(0x3ff));

        __mmask32 mask_ge = _mm512_cmp_epu16_mask(
                sign_x, sign_y, _MM_CMPINT_LT); // only greater than
        __mmask32 sign_eq = _mm512_cmpeq_epu16_mask(sign_x, sign_y);
        __mmask32 neg = _mm512_mask_cmpeq_epu16_mask(
                sign_eq,
                sign_x,
                _mm512_set1_epi16(0x8000)); // both numbers are -ve

        // compare exponents only if signs are equal:
        mask_ge = mask_ge
                | _mm512_mask_cmp_epu16_mask(
                          sign_eq, exp_x, exp_y, _MM_CMPINT_NLE);
        // get mask for elements for which both sign and exponents are equal:
        __mmask32 exp_eq = _mm512_mask_cmpeq_epu16_mask(sign_eq, exp_x, exp_y);

        // compare mantissa for elements for which both sign and expponent are equal:
        mask_ge = mask_ge
                | _mm512_mask_cmp_epu16_mask(
                          exp_eq, mant_x, mant_y, _MM_CMPINT_NLT);
        return _kxor_mask32(mask_ge, neg);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_mask_mov_epi16(y, ge(x, y), x);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, ge(x, y), y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    // Apparently this is a terrible for perf, npy_half_to_float seems to work
    // better
    //static float uint16_to_float(uint16_t val)
    //{
    //    // Ideally use _mm_loadu_si16, but its only gcc > 11.x
    //    // TODO: use inline ASM? https://godbolt.org/z/aGYvh7fMM
    //    __m128i xmm = _mm_maskz_loadu_epi16(0x01, &val);
    //    __m128 xmm2 = _mm_cvtph_ps(xmm);
    //    return _mm_cvtss_f32(xmm2);
    //}
    static type_t float_to_uint16(float val)
    {
        __m128 xmm = _mm_load_ss(&val);
        __m128i xmm2 = _mm_cvtps_ph(xmm, _MM_FROUND_NO_EXC);
        return _mm_extract_epi16(xmm2, 0);
    }
    static type_t reducemax(zmm_t v)
    {
        __m512 lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 0));
        __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 1));
        float lo_max = _mm512_reduce_max_ps(lo);
        float hi_max = _mm512_reduce_max_ps(hi);
        return float_to_uint16(std::max(lo_max, hi_max));
    }
    static type_t reducemin(zmm_t v)
    {
        __m512 lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 0));
        __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 1));
        float lo_max = _mm512_reduce_min_ps(lo);
        float hi_max = _mm512_reduce_min_ps(hi);
        return float_to_uint16(std::min(lo_max, hi_max));
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
struct zmm_vector<int16_t> {
    using type_t = int16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT16;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT16;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }

    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi16_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
struct zmm_vector<uint16_t> {
    using type_t = uint16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT16;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu16_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

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
};

template <>
struct zmm_vector<int64_t> {
    using type_t = int64_t;
    using zmm_t = __m512i;
    using ymm_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT64;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT64;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    } // TODO: this should broadcast bits as is?

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_EQ);
    }
    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi64(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epi64(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epi64(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
struct zmm_vector<uint64_t> {
    using type_t = uint64_t;
    using zmm_t = __m512i;
    using ymm_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT64;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    }

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t eq(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_EQ);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu64(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epu64(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epu64(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
struct zmm_vector<double> {
    using type_t = double;
    using zmm_t = __m512d;
    using ymm_t = __m512d;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITY;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITY;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_pd(type_max());
    }

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_pd(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_GE_OQ);
    }
    static opmask_t eq(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
    }
    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_pd(index, base, scale);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_pd(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_pd(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_pd(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_pd(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_pd(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_pd(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_pd(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_pd(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_pd(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_pd(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_pd(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_pd(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_pd(mem, x);
    }
};

#endif //AVX512_ZMM_CLASSES
