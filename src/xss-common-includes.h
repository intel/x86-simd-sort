#ifndef XSS_COMMON_INCLUDES
#define XSS_COMMON_INCLUDES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>
#include <thread>
#include <mutex>
#include "xss-custom-float.h"

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

#define PRAGMA(x) _Pragma(#x)
#define UNUSED(x) (void)(x)

/* Compiler specific macros specific */
#ifdef _MSC_VER
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static __forceinline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#elif defined(__CYGWIN__)
/*
 * Force inline in cygwin to work around a compiler bug. See
 * https://github.com/numpy/numpy/pull/22315#issuecomment-1267757584
 */
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static __attribute__((always_inline))
#define X86_SIMD_SORT_FINLINE static __attribute__((always_inline))
#elif defined(__GNUC__)
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#else
#define X86_SIMD_SORT_INLINE_ONLY
#define X86_SIMD_SORT_INLINE static
#define X86_SIMD_SORT_FINLINE static
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#if defined(__INTEL_COMPILER) and !defined(__SANITIZE_ADDRESS__)
#define X86_SIMD_SORT_UNROLL_LOOP(num) PRAGMA(unroll(num))
#elif __GNUC__ >= 8 and !defined(__SANITIZE_ADDRESS__)
#define X86_SIMD_SORT_UNROLL_LOOP(num) PRAGMA(GCC unroll num)
#else
#define X86_SIMD_SORT_UNROLL_LOOP(num)
#endif

#define NETWORK_REVERSE_4LANES 0, 1, 2, 3
#define NETWORK_REVERSE_8LANES 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_REVERSE_16LANES \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define NETWORK_REVERSE_32LANES \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31

#if defined(XSS_USE_OPENMP) && defined(_OPENMP)
#define XSS_COMPILE_OPENMP
#include <omp.h>
#endif

struct float16 {
    uint16_t val;
};

template <class... T>
constexpr bool always_false = false;

typedef size_t arrsize_t;

template <typename type>
struct zmm_vector;

template <typename type>
struct ymm_vector;

template <typename type>
struct avx2_vector;

template <typename type>
struct avx2_half_vector;

enum class simd_type : int { AVX2, AVX512 };

template <typename vtype, typename T = typename vtype::type_t>
X86_SIMD_SORT_INLINE bool comparison_func(const T &a, const T &b);

struct threadmanager {
    int max_thread_count;
    std::mutex mymutex;
    int sharedCount;
    arrsize_t task_threshold;

    threadmanager() {
#ifdef XSS_COMPILE_OPENMP
        max_thread_count = 8;
#else
        max_thread_count = 0;
#endif
        sharedCount = 0;
        task_threshold = 100000;
    };
    void incrementCount(int ii) {
        mymutex.lock();
        sharedCount += ii;
        mymutex.unlock();
    }
};

#endif // XSS_COMMON_INCLUDES
