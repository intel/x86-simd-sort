/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * Copyright (C) 2021 Serge Sans Paille
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 *          Serge Sans Paille <serge.guelton@telecom-bretagne.eu>
 *          Liu Zhuan <zhuan.liu@intel.com>
 *          Tang Xi <xi.tang@intel.com>
 * ****************************************************************/

#ifndef AVX512_QSORT_COMMON
#define AVX512_QSORT_COMMON

/*
 * Quicksort using AVX-512. The ideas and code are based on these two research
 * papers [1] and [2]. On a high level, the idea is to vectorize quicksort
 * partitioning using AVX-512 compressstore instructions. If the array size is
 * < 128, then use Bitonic sorting network implemented on 512-bit registers.
 * The precise network definitions depend on the dtype and are defined in
 * separate files: avx512-16bit-qsort.hpp, avx512-32bit-qsort.hpp and
 * avx512-64bit-qsort.hpp. Article [4] is a good resource for bitonic sorting
 * network. The core implementations of the vectorized qsort functions
 * avx512_qsort<T>(T*, int64_t) are modified versions of avx2 quicksort
 * presented in the paper [2] and source code associated with that paper [3].
 *
 * [1] Fast and Robust Vectorized In-Place Sorting of Primitive Types
 *     https://drops.dagstuhl.de/opus/volltexte/2021/13775/
 *
 * [2] A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
 * Skylake https://arxiv.org/pdf/1704.08579.pdf
 *
 * [3] https://github.com/simd-sorting/fast-and-robust: SPDX-License-Identifier: MIT
 *
 * [4] http://mitp-content-server.mit.edu:18180/books/content/sectbyfn?collid=books_pres_0&fn=Chapter%2027.pdf&id=8030
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>

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

#define PRAGMA(x) _Pragma (#x)

/* Compiler specific macros specific */
#ifdef _MSC_VER
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static __forceinline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
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
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#else
#define X86_SIMD_SORT_INLINE static
#define X86_SIMD_SORT_FINLINE static
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#if __GNUC__ >= 8
#define X86_SIMD_SORT_UNROLL_LOOP(num) PRAGMA(GCC unroll num)
#else
#define X86_SIMD_SORT_UNROLL_LOOP(num)
#endif

template <typename type>
struct zmm_vector;

template <typename type>
struct ymm_vector;

template <typename T>
bool is_a_nan(T elem)
{
    return std::isnan(elem);
}

template <typename vtype, typename T>
int64_t replace_nan_with_inf(T *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    using opmask_t = typename vtype::opmask_t;
    using reg_t = typename vtype::reg_t;
    opmask_t loadmask;
    reg_t in;
    while (arrsize > 0) {
        if (arrsize < vtype::numlanes) {
            loadmask = vtype::get_partial_loadmask(arrsize);
            in = vtype::maskz_loadu(loadmask, arr);
        }
        else {
            in = vtype::loadu(arr);
        }
        opmask_t nanmask = vtype::template fpclass<0x01 | 0x80>(in);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        vtype::mask_storeu(arr, nanmask, vtype::zmm_max());
        arr += vtype::numlanes;
        arrsize -= vtype::numlanes;
    }
    return nan_count;
}

template <typename vtype, typename type_t>
bool has_nan(type_t *arr, int64_t arrsize)
{
    using opmask_t = typename vtype::opmask_t;
    using reg_t = typename vtype::reg_t;
    bool found_nan = false;
    opmask_t loadmask;
    reg_t in;
    while (arrsize > 0) {
        if (arrsize < vtype::numlanes) {
            loadmask = vtype::get_partial_loadmask(arrsize);
            in = vtype::maskz_loadu(loadmask, arr);
        }
        else {
            in = vtype::loadu(arr);
        }
        opmask_t nanmask = vtype::template fpclass<0x01 | 0x80>(in);
        arr += vtype::numlanes;
        arrsize -= vtype::numlanes;
        if (nanmask != 0x00) {
            found_nan = true;
            break;
        }
    }
    return found_nan;
}

template <typename type_t>
void replace_inf_with_nan(type_t *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        if constexpr (std::is_floating_point_v<type_t>) {
            arr[ii] = std::numeric_limits<type_t>::quiet_NaN();
        }
        else {
            arr[ii] = 0xFFFF;
        }
        nan_count -= 1;
    }
}

/*
 * Sort all the NAN's to end of the array and return the index of the last elem
 * in the array which is not a nan
 */
template <typename T>
int64_t move_nans_to_end_of_array(T *arr, int64_t arrsize)
{
    int64_t jj = arrsize - 1;
    int64_t ii = 0;
    int64_t count = 0;
    while (ii <= jj) {
        if (is_a_nan(arr[ii])) {
            std::swap(arr[ii], arr[jj]);
            jj -= 1;
            count++;
        }
        else {
            ii += 1;
        }
    }
    return arrsize - count - 1;
}

template <typename vtype, typename T = typename vtype::type_t>
bool comparison_func(const T &a, const T &b)
{
    return a < b;
}

/*
 * COEX == Compare and Exchange two registers by swapping min and max values
 */
template <typename vtype, typename mm_t>
static void COEX(mm_t &a, mm_t &b)
{
    mm_t temp = a;
    a = vtype::min(a, b);
    b = vtype::max(temp, b);
}
template <typename vtype,
          typename reg_t = typename vtype::reg_t,
          typename opmask_t = typename vtype::opmask_t>
static inline reg_t cmp_merge(reg_t in1, reg_t in2, opmask_t mask)
{
    reg_t min = vtype::min(in2, in1);
    reg_t max = vtype::max(in2, in1);
    return vtype::mask_mov(min, mask, max); // 0 -> min, 1 -> max
}
/*
 * Parition one ZMM register based on the pivot and returns the
 * number of elements that are greater than or equal to the pivot.
 */
template <typename vtype, typename type_t, typename reg_t>
static inline int32_t partition_vec(type_t *arr,
                                    int64_t left,
                                    int64_t right,
                                    const reg_t curr_vec,
                                    const reg_t pivot_vec,
                                    reg_t *smallest_vec,
                                    reg_t *biggest_vec)
{
    /* which elements are larger than or equal to the pivot */
    typename vtype::opmask_t ge_mask = vtype::ge(curr_vec, pivot_vec);
    int32_t amount_ge_pivot = _mm_popcnt_u32((int32_t)ge_mask);
    vtype::mask_compressstoreu(
            arr + left, vtype::knot_opmask(ge_mask), curr_vec);
    vtype::mask_compressstoreu(
            arr + right - amount_ge_pivot, ge_mask, curr_vec);
    *smallest_vec = vtype::min(curr_vec, *smallest_vec);
    *biggest_vec = vtype::max(curr_vec, *biggest_vec);
    return amount_ge_pivot;
}
/*
 * Parition an array based on the pivot and returns the index of the
 * first element that is greater than or equal to the pivot.
 */
template <typename vtype, typename type_t>
static inline int64_t partition_avx512(type_t *arr,
                                       int64_t left,
                                       int64_t right,
                                       type_t pivot,
                                       type_t *smallest,
                                       type_t *biggest)
{
    /* make array length divisible by vtype::numlanes , shortening the array */
    for (int32_t i = (right - left) % vtype::numlanes; i > 0; --i) {
        *smallest = std::min(*smallest, arr[left], comparison_func<vtype>);
        *biggest = std::max(*biggest, arr[left], comparison_func<vtype>);
        if (!comparison_func<vtype>(arr[left], pivot)) {
            std::swap(arr[left], arr[--right]);
        }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than vtype::numlanes elements in the array */

    using reg_t = typename vtype::reg_t;
    reg_t pivot_vec = vtype::set1(pivot);
    reg_t min_vec = vtype::set1(*smallest);
    reg_t max_vec = vtype::set1(*biggest);

    if (right - left == vtype::numlanes) {
        reg_t vec = vtype::loadu(arr + left);
        int32_t amount_ge_pivot = partition_vec<vtype>(arr,
                                                       left,
                                                       left + vtype::numlanes,
                                                       vec,
                                                       pivot_vec,
                                                       &min_vec,
                                                       &max_vec);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return left + (vtype::numlanes - amount_ge_pivot);
    }

    // first and last vtype::numlanes values are partitioned at the end
    reg_t vec_left = vtype::loadu(arr + left);
    reg_t vec_right = vtype::loadu(arr + (right - vtype::numlanes));
    // store points of the vectors
    int64_t r_store = right - vtype::numlanes;
    int64_t l_store = left;
    // indices for loading the elements
    left += vtype::numlanes;
    right -= vtype::numlanes;
    while (right - left != 0) {
        reg_t curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + vtype::numlanes) - right < left - l_store) {
            right -= vtype::numlanes;
            curr_vec = vtype::loadu(arr + right);
        }
        else {
            curr_vec = vtype::loadu(arr + left);
            left += vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        int32_t amount_ge_pivot
                = partition_vec<vtype>(arr,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       curr_vec,
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        ;
        r_store -= amount_ge_pivot;
        l_store += (vtype::numlanes - amount_ge_pivot);
    }

    /* partition and save vec_left and vec_right */
    int32_t amount_ge_pivot = partition_vec<vtype>(arr,
                                                   l_store,
                                                   r_store + vtype::numlanes,
                                                   vec_left,
                                                   pivot_vec,
                                                   &min_vec,
                                                   &max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    amount_ge_pivot = partition_vec<vtype>(arr,
                                           l_store,
                                           l_store + vtype::numlanes,
                                           vec_right,
                                           pivot_vec,
                                           &min_vec,
                                           &max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype,
          int num_unroll,
          typename type_t = typename vtype::type_t>
static inline int64_t partition_avx512_unrolled(type_t *arr,
                                                int64_t left,
                                                int64_t right,
                                                type_t pivot,
                                                type_t *smallest,
                                                type_t *biggest)
{
    if constexpr (num_unroll == 0) {
        return partition_avx512<vtype>(
                arr, left, right, pivot, smallest, biggest);
    }

    if (right - left <= 2 * num_unroll * vtype::numlanes) {
        return partition_avx512<vtype>(
                arr, left, right, pivot, smallest, biggest);
    }
    /* make array length divisible by 8*vtype::numlanes , shortening the array */
    for (int32_t i = ((right - left) % (num_unroll * vtype::numlanes)); i > 0;
         --i) {
        *smallest = std::min(*smallest, arr[left], comparison_func<vtype>);
        *biggest = std::max(*biggest, arr[left], comparison_func<vtype>);
        if (!comparison_func<vtype>(arr[left], pivot)) {
            std::swap(arr[left], arr[--right]);
        }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than vtype::numlanes elements in the array */

    using reg_t = typename vtype::reg_t;
    reg_t pivot_vec = vtype::set1(pivot);
    reg_t min_vec = vtype::set1(*smallest);
    reg_t max_vec = vtype::set1(*biggest);

    // We will now have atleast 16 registers worth of data to process:
    // left and right vtype::numlanes values are partitioned at the end
    reg_t vec_left[num_unroll], vec_right[num_unroll];
X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        vec_left[ii] = vtype::loadu(arr + left + vtype::numlanes * ii);
        vec_right[ii] = vtype::loadu(
                arr + (right - vtype::numlanes * (num_unroll - ii)));
    }
    // store points of the vectors
    int64_t r_store = right - vtype::numlanes;
    int64_t l_store = left;
    // indices for loading the elements
    left += num_unroll * vtype::numlanes;
    right -= num_unroll * vtype::numlanes;
    while (right - left != 0) {
        reg_t curr_vec[num_unroll];
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + vtype::numlanes) - right < left - l_store) {
            right -= num_unroll * vtype::numlanes;
X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + right + ii * vtype::numlanes);
            }
        }
        else {
X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + left + ii * vtype::numlanes);
            }
            left += num_unroll * vtype::numlanes;
        }
// partition the current vector and save it on both sides of the array
X86_SIMD_SORT_UNROLL_LOOP(8)
        for (int ii = 0; ii < num_unroll; ++ii) {
            int32_t amount_ge_pivot
                    = partition_vec<vtype>(arr,
                                           l_store,
                                           r_store + vtype::numlanes,
                                           curr_vec[ii],
                                           pivot_vec,
                                           &min_vec,
                                           &max_vec);
            l_store += (vtype::numlanes - amount_ge_pivot);
            r_store -= amount_ge_pivot;
        }
    }

/* partition and save vec_left[8] and vec_right[8] */
X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        int32_t amount_ge_pivot
                = partition_vec<vtype>(arr,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       vec_left[ii],
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        r_store -= amount_ge_pivot;
    }
X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        int32_t amount_ge_pivot
                = partition_vec<vtype>(arr,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       vec_right[ii],
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        r_store -= amount_ge_pivot;
    }
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

// Key-value sort helper functions

template <typename vtype1,
          typename vtype2,
          typename reg_t1 = typename vtype1::reg_t,
          typename reg_t2 = typename vtype2::reg_t>
static void COEX(reg_t1 &key1, reg_t1 &key2, reg_t2 &index1, reg_t2 &index2)
{
    reg_t1 key_t1 = vtype1::min(key1, key2);
    reg_t1 key_t2 = vtype1::max(key1, key2);

    reg_t2 index_t1
            = vtype2::mask_mov(index2, vtype1::eq(key_t1, key1), index1);
    reg_t2 index_t2
            = vtype2::mask_mov(index1, vtype1::eq(key_t1, key1), index2);

    key1 = key_t1;
    key2 = key_t2;
    index1 = index_t1;
    index2 = index_t2;
}
template <typename vtype1,
          typename vtype2,
          typename reg_t1 = typename vtype1::reg_t,
          typename reg_t2 = typename vtype2::reg_t,
          typename opmask_t = typename vtype1::opmask_t>
static inline reg_t1 cmp_merge(reg_t1 in1,
                               reg_t1 in2,
                               reg_t2 &indexes1,
                               reg_t2 indexes2,
                               opmask_t mask)
{
    reg_t1 tmp_keys = cmp_merge<vtype1>(in1, in2, mask);
    indexes1 = vtype2::mask_mov(indexes2, vtype1::eq(tmp_keys, in1), indexes1);
    return tmp_keys; // 0 -> min, 1 -> max
}

/*
 * Parition one ZMM register based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype1,
          typename vtype2,
          typename type_t1 = typename vtype1::type_t,
          typename type_t2 = typename vtype2::type_t,
          typename reg_t1 = typename vtype1::reg_t,
          typename reg_t2 = typename vtype2::reg_t>
static inline int32_t partition_vec(type_t1 *keys,
                                    type_t2 *indexes,
                                    int64_t left,
                                    int64_t right,
                                    const reg_t1 keys_vec,
                                    const reg_t2 indexes_vec,
                                    const reg_t1 pivot_vec,
                                    reg_t1 *smallest_vec,
                                    reg_t1 *biggest_vec)
{
    /* which elements are larger than the pivot */
    typename vtype1::opmask_t gt_mask = vtype1::ge(keys_vec, pivot_vec);
    int32_t amount_gt_pivot = _mm_popcnt_u32((int32_t)gt_mask);
    vtype1::mask_compressstoreu(
            keys + left, vtype1::knot_opmask(gt_mask), keys_vec);
    vtype1::mask_compressstoreu(
            keys + right - amount_gt_pivot, gt_mask, keys_vec);
    vtype2::mask_compressstoreu(
            indexes + left, vtype2::knot_opmask(gt_mask), indexes_vec);
    vtype2::mask_compressstoreu(
            indexes + right - amount_gt_pivot, gt_mask, indexes_vec);
    *smallest_vec = vtype1::min(keys_vec, *smallest_vec);
    *biggest_vec = vtype1::max(keys_vec, *biggest_vec);
    return amount_gt_pivot;
}
/*
 * Parition an array based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype1,
          typename vtype2,
          typename type_t1 = typename vtype1::type_t,
          typename type_t2 = typename vtype2::type_t,
          typename reg_t1 = typename vtype1::reg_t,
          typename reg_t2 = typename vtype2::reg_t>
static inline int64_t partition_avx512(type_t1 *keys,
                                       type_t2 *indexes,
                                       int64_t left,
                                       int64_t right,
                                       type_t1 pivot,
                                       type_t1 *smallest,
                                       type_t1 *biggest)
{
    /* make array length divisible by vtype1::numlanes , shortening the array */
    for (int32_t i = (right - left) % vtype1::numlanes; i > 0; --i) {
        *smallest = std::min(*smallest, keys[left]);
        *biggest = std::max(*biggest, keys[left]);
        if (keys[left] > pivot) {
            right--;
            std::swap(keys[left], keys[right]);
            std::swap(indexes[left], indexes[right]);
        }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than vtype1::numlanes elements in the array */

    reg_t1 pivot_vec = vtype1::set1(pivot);
    reg_t1 min_vec = vtype1::set1(*smallest);
    reg_t1 max_vec = vtype1::set1(*biggest);

    if (right - left == vtype1::numlanes) {
        reg_t1 keys_vec = vtype1::loadu(keys + left);
        int32_t amount_gt_pivot;

        reg_t2 indexes_vec = vtype2::loadu(indexes + left);
        amount_gt_pivot = partition_vec<vtype1, vtype2>(keys,
                                                        indexes,
                                                        left,
                                                        left + vtype1::numlanes,
                                                        keys_vec,
                                                        indexes_vec,
                                                        pivot_vec,
                                                        &min_vec,
                                                        &max_vec);

        *smallest = vtype1::reducemin(min_vec);
        *biggest = vtype1::reducemax(max_vec);
        return left + (vtype1::numlanes - amount_gt_pivot);
    }

    // first and last vtype1::numlanes values are partitioned at the end
    reg_t1 keys_vec_left = vtype1::loadu(keys + left);
    reg_t1 keys_vec_right = vtype1::loadu(keys + (right - vtype1::numlanes));
    reg_t2 indexes_vec_left;
    reg_t2 indexes_vec_right;
    indexes_vec_left = vtype2::loadu(indexes + left);
    indexes_vec_right = vtype2::loadu(indexes + (right - vtype1::numlanes));

    // store points of the vectors
    int64_t r_store = right - vtype1::numlanes;
    int64_t l_store = left;
    // indices for loading the elements
    left += vtype1::numlanes;
    right -= vtype1::numlanes;
    while (right - left != 0) {
        reg_t1 keys_vec;
        reg_t2 indexes_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + vtype1::numlanes) - right < left - l_store) {
            right -= vtype1::numlanes;
            keys_vec = vtype1::loadu(keys + right);
            indexes_vec = vtype2::loadu(indexes + right);
        }
        else {
            keys_vec = vtype1::loadu(keys + left);
            indexes_vec = vtype2::loadu(indexes + left);
            left += vtype1::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        int32_t amount_gt_pivot;

        amount_gt_pivot
                = partition_vec<vtype1, vtype2>(keys,
                                                indexes,
                                                l_store,
                                                r_store + vtype1::numlanes,
                                                keys_vec,
                                                indexes_vec,
                                                pivot_vec,
                                                &min_vec,
                                                &max_vec);
        r_store -= amount_gt_pivot;
        l_store += (vtype1::numlanes - amount_gt_pivot);
    }

    /* partition and save vec_left and vec_right */
    int32_t amount_gt_pivot;
    amount_gt_pivot = partition_vec<vtype1, vtype2>(keys,
                                                    indexes,
                                                    l_store,
                                                    r_store + vtype1::numlanes,
                                                    keys_vec_left,
                                                    indexes_vec_left,
                                                    pivot_vec,
                                                    &min_vec,
                                                    &max_vec);
    l_store += (vtype1::numlanes - amount_gt_pivot);
    amount_gt_pivot = partition_vec<vtype1, vtype2>(keys,
                                                    indexes,
                                                    l_store,
                                                    l_store + vtype1::numlanes,
                                                    keys_vec_right,
                                                    indexes_vec_right,
                                                    pivot_vec,
                                                    &min_vec,
                                                    &max_vec);
    l_store += (vtype1::numlanes - amount_gt_pivot);
    *smallest = vtype1::reducemin(min_vec);
    *biggest = vtype1::reducemax(max_vec);
    return l_store;
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_scalar(type_t *arr,
                                      const int64_t left,
                                      const int64_t right)
{
    constexpr int64_t numSamples = vtype::numlanes;
    type_t samples[numSamples];

    int64_t delta = (right - left) / numSamples;

    for (int i = 0; i < numSamples; i++) {
        samples[i] = arr[left + i * delta];
    }

    auto vec = vtype::loadu(samples);
    vec = vtype::sort_vec(vec);
    return ((type_t *)&vec)[numSamples / 2];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_16bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 32
    int64_t size = (right - left) / 32;
    type_t vec_arr[32] = {arr[left],
                          arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size],
                          arr[left + 17 * size],
                          arr[left + 18 * size],
                          arr[left + 19 * size],
                          arr[left + 20 * size],
                          arr[left + 21 * size],
                          arr[left + 22 * size],
                          arr[left + 23 * size],
                          arr[left + 24 * size],
                          arr[left + 25 * size],
                          arr[left + 26 * size],
                          arr[left + 27 * size],
                          arr[left + 28 * size],
                          arr[left + 29 * size],
                          arr[left + 30 * size],
                          arr[left + 31 * size]};
    typename vtype::reg_t rand_vec = vtype::loadu(vec_arr);
    typename vtype::reg_t sort = vtype::sort_vec(rand_vec);
    return ((type_t *)&sort)[16];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_32bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 16
    int64_t size = (right - left) / 16;
    using reg_t = typename vtype::reg_t;
    type_t vec_arr[16] = {arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size]};
    reg_t rand_vec = vtype::loadu(vec_arr);
    reg_t sort = vtype::sort_vec(rand_vec);
    // pivot will never be a nan, since there are no nan's!
    return ((type_t *)&sort)[8];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_64bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 8
    int64_t size = (right - left) / 8;
    using reg_t = typename vtype::reg_t;
    reg_t rand_vec = vtype::set(arr[left + size],
                                arr[left + 2 * size],
                                arr[left + 3 * size],
                                arr[left + 4 * size],
                                arr[left + 5 * size],
                                arr[left + 6 * size],
                                arr[left + 7 * size],
                                arr[left + 8 * size]);
    // pivot will never be a nan, since there are no nan's!
    reg_t sort = vtype::sort_vec(rand_vec);
    return ((type_t *)&sort)[4];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot(type_t *arr,
                                      const int64_t left,
                                      const int64_t right)
{
    if constexpr (vtype::numlanes == 8)
        return get_pivot_64bit<vtype>(arr, left, right);
    else if constexpr (vtype::numlanes == 16)
        return get_pivot_32bit<vtype>(arr, left, right);
    else if constexpr (vtype::numlanes == 32)
        return get_pivot_16bit<vtype>(arr, left, right);
    else
        return get_pivot_scalar<vtype>(arr, left, right);
}

template <typename vtype, int64_t maxN>
X86_SIMD_SORT_INLINE void sort_n(typename vtype::type_t *arr, int N);

template <typename vtype, typename type_t>
static void qsort_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= vtype::network_sort_threshold
     */
    if (right + 1 - left <= vtype::network_sort_threshold) {
        sort_n<vtype, vtype::network_sort_threshold>(
                arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot<vtype, type_t>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();

    int64_t pivot_index
            = partition_avx512_unrolled<vtype, vtype::partition_unroll_factor>(
                    arr, left, right + 1, pivot, &smallest, &biggest);

    if (pivot != smallest)
        qsort_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest) qsort_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
static void qselect_(type_t *arr,
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
     * Base case: use bitonic networks to sort arrays <= vtype::network_sort_threshold
     */
    if (right + 1 - left <= vtype::network_sort_threshold) {
        sort_n<vtype, vtype::network_sort_threshold>(
                arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot<vtype, type_t>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();

    int64_t pivot_index
            = partition_avx512_unrolled<vtype, vtype::partition_unroll_factor>(
                    arr, left, right + 1, pivot, &smallest, &biggest);

    if ((pivot != smallest) && (pos < pivot_index))
        qselect_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

// Regular quicksort routines:
template <typename T>
void avx512_qsort(T *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
        if constexpr (std::is_floating_point_v<T>) {
            int64_t nan_count
                    = replace_nan_with_inf<zmm_vector<T>>(arr, arrsize);
            qsort_<zmm_vector<T>, T>(
                    arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
            replace_inf_with_nan(arr, arrsize, nan_count);
        }
        else {
            qsort_<zmm_vector<T>, T>(
                    arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        }
    }
}

void avx512_qsort_fp16(uint16_t *arr, int64_t arrsize);

template <typename T>
void avx512_qselect(T *arr, int64_t k, int64_t arrsize, bool hasnan = false)
{
    int64_t indx_last_elem = arrsize - 1;
    /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
    if constexpr (std::is_floating_point_v<T>) {
        if (UNLIKELY(hasnan)) {
            indx_last_elem = move_nans_to_end_of_array(arr, arrsize);
        }
    }
    if (indx_last_elem >= k) {
        qselect_<zmm_vector<T>, T>(
                arr, k, 0, indx_last_elem, 2 * (int64_t)log2(indx_last_elem));
    }
}

void avx512_qselect_fp16(uint16_t *arr,
                         int64_t k,
                         int64_t arrsize,
                         bool hasnan = false);

template <typename T>
inline void
avx512_partial_qsort(T *arr, int64_t k, int64_t arrsize, bool hasnan = false)
{
    avx512_qselect<T>(arr, k - 1, arrsize, hasnan);
    avx512_qsort<T>(arr, k - 1);
}
inline void avx512_partial_qsort_fp16(uint16_t *arr,
                                      int64_t k,
                                      int64_t arrsize,
                                      bool hasnan = false)
{
    avx512_qselect_fp16(arr, k - 1, arrsize, hasnan);
    avx512_qsort_fp16(arr, k - 1);
}

#endif // AVX512_QSORT_COMMON
