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
 * avx512_qsort<T>(T*, arrsize_t) are modified versions of avx2 quicksort
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
#include "xss-common-includes.h"
#include "xss-pivot-selection.hpp"
#include "xss-network-qsort.hpp"

namespace xss{
namespace avx2{
template <typename type>
struct ymm_vector;

inline int64_t replace_nan_with_inf(float *arr, int64_t arrsize);
}
}

// key-value sort routines
template <typename T1, typename T2>
void avx512_qsort_kv(T1 *keys, T2 *indexes, int64_t arrsize);

template <typename T>
bool is_a_nan(T elem)
{
    return std::isnan(elem);
}

template <typename vtype, typename T>
X86_SIMD_SORT_INLINE arrsize_t replace_nan_with_inf(T *arr, arrsize_t size)
{
    arrsize_t nan_count = 0;
    using opmask_t = typename vtype::opmask_t;
    using reg_t = typename vtype::reg_t;
    opmask_t loadmask;
    reg_t in;
    /*
     * (ii + numlanes) can never overflow: max val of size is 2**63 on 64-bit
     * and 2**31 on 32-bit systems
     */
    for (arrsize_t ii = 0; ii < size; ii = ii + vtype::numlanes) {
        if (size - ii < vtype::numlanes) {
            loadmask = vtype::get_partial_loadmask(size - ii);
            in = vtype::maskz_loadu(loadmask, arr + ii);
        }
        else {
            in = vtype::loadu(arr + ii);
        }
        opmask_t nanmask = vtype::template fpclass<0x01 | 0x80>(in);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        vtype::mask_storeu(arr + ii, nanmask, vtype::zmm_max());
    }
    return nan_count;
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE bool has_nan(type_t *arr, arrsize_t size)
{
    using opmask_t = typename vtype::opmask_t;
    using reg_t = typename vtype::reg_t;
    bool found_nan = false;
    opmask_t loadmask;
    reg_t in;
    for (arrsize_t ii = 0; ii < size; ii = ii + vtype::numlanes) {
        if (size - ii < vtype::numlanes) {
            loadmask = vtype::get_partial_loadmask(size - ii);
            in = vtype::maskz_loadu(loadmask, arr + ii);
        }
        else {
            in = vtype::loadu(arr + ii);
        }
        opmask_t nanmask = vtype::template fpclass<0x01 | 0x80>(in);
        if (nanmask != 0x00) {
            found_nan = true;
            break;
        }
    }
    return found_nan;
}

template <typename type_t>
X86_SIMD_SORT_INLINE void
replace_inf_with_nan(type_t *arr, arrsize_t size, arrsize_t nan_count)
{
    for (arrsize_t ii = size - 1; nan_count > 0; --ii) {
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
X86_SIMD_SORT_INLINE arrsize_t move_nans_to_end_of_array(T *arr, arrsize_t size)
{
    arrsize_t jj = size - 1;
    arrsize_t ii = 0;
    arrsize_t count = 0;
    while (ii < jj) {
        if (is_a_nan(arr[ii])) {
            std::swap(arr[ii], arr[jj]);
            jj -= 1;
            count++;
        }
        else {
            ii += 1;
        }
    }
    /* Haven't checked for nan when ii == jj */
    if (is_a_nan(arr[ii])) { count++; }
    return size - count - 1;
}

template <typename vtype, typename T = typename vtype::type_t>
X86_SIMD_SORT_INLINE bool comparison_func(const T &a, const T &b)
{
    return a < b;
}

/*
 * COEX == Compare and Exchange two registers by swapping min and max values
 */
template <typename vtype, typename mm_t>
X86_SIMD_SORT_INLINE void COEX(mm_t &a, mm_t &b)
{
    mm_t temp = a;
    a = vtype::min(a, b);
    b = vtype::max(temp, b);
}

template <typename vtype,
          typename reg_t = typename vtype::reg_t,
          typename opmask_t = typename vtype::opmask_t>
X86_SIMD_SORT_INLINE reg_t cmp_merge(reg_t in1, reg_t in2, opmask_t mask)
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
X86_SIMD_SORT_INLINE arrsize_t partition_vec_avx512(type_t *l_store,
                                             type_t *r_store,
                                             const reg_t curr_vec,
                                             const reg_t pivot_vec,
                                             reg_t &smallest_vec,
                                             reg_t &biggest_vec)
{
    typename vtype::opmask_t ge_mask = vtype::ge(curr_vec, pivot_vec);
    int amount_ge_pivot = _mm_popcnt_u32((int)ge_mask);

    vtype::mask_compressstoreu(l_store, vtype::knot_opmask(ge_mask), curr_vec);
    vtype::mask_compressstoreu(
            r_store + vtype::numlanes - amount_ge_pivot, ge_mask, curr_vec);

    smallest_vec = vtype::min(curr_vec, smallest_vec);
    biggest_vec = vtype::max(curr_vec, biggest_vec);

    return amount_ge_pivot;
}
/*
 * Parition one YMM register based on the pivot and returns the
 * number of elements that are greater than or equal to the pivot.
 */
template <typename vtype, typename type_t, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE arrsize_t partition_vec_avx2(type_t *l_store,
                                             type_t *r_store,
                                             const reg_t curr_vec,
                                             const reg_t pivot_vec,
                                             reg_t &smallest_vec,
                                             reg_t &biggest_vec)
{
    /* which elements are larger than or equal to the pivot */
    typename vtype::opmask_t ge_mask = vtype::ge(curr_vec, pivot_vec);

    int32_t amount_ge_pivot = vtype::double_compressstore(
            l_store, r_store, ge_mask, curr_vec);

    smallest_vec = vtype::min(curr_vec, smallest_vec);
    biggest_vec = vtype::max(curr_vec, biggest_vec);

    return amount_ge_pivot;
}

// Generic function dispatches to AVX2 or AVX512 code
template <typename vtype, typename type_t, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE arrsize_t partition_vec(type_t *l_store,
                                             type_t *r_store,
                                             const reg_t curr_vec,
                                             const reg_t pivot_vec,
                                             reg_t &smallest_vec,
                                             reg_t &biggest_vec)
{
    if constexpr (sizeof(reg_t) == 64){
        return partition_vec_avx512<vtype>(l_store, r_store, curr_vec, pivot_vec, smallest_vec, biggest_vec);
    }else if constexpr (sizeof(reg_t) == 32){
        return partition_vec_avx2<vtype>(l_store, r_store, curr_vec, pivot_vec, smallest_vec, biggest_vec);
    }else{
        static_assert(sizeof(reg_t) == -1, "should not reach here");
        return 0;
    }
}

/*
 * Parition an array based on the pivot and returns the index of the
 * first element that is greater than or equal to the pivot.
 */
template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE arrsize_t partition_avx512(type_t *arr,
                                                arrsize_t left,
                                                arrsize_t right,
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
        arrsize_t unpartitioned = right - left - vtype::numlanes;
        arrsize_t l_store = left;

        arrsize_t amount_ge_pivot
                = partition_vec<vtype>(arr + l_store,
                                       arr + l_store + unpartitioned,
                                       vec,
                                       pivot_vec,
                                       min_vec,
                                       max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return l_store;
    }

    // first and last vtype::numlanes values are partitioned at the end
    reg_t vec_left = vtype::loadu(arr + left);
    reg_t vec_right = vtype::loadu(arr + (right - vtype::numlanes));
    // store points of the vectors
    arrsize_t unpartitioned = right - left - vtype::numlanes;
    arrsize_t l_store = left;
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
        if ((l_store + unpartitioned + vtype::numlanes) - right
            < left - l_store) {
            right -= vtype::numlanes;
            curr_vec = vtype::loadu(arr + right);
        }
        else {
            curr_vec = vtype::loadu(arr + left);
            left += vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        arrsize_t amount_ge_pivot
                = partition_vec<vtype>(arr + l_store,
                                       arr + l_store + unpartitioned,
                                       curr_vec,
                                       pivot_vec,
                                       min_vec,
                                       max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    /* partition and save vec_left and vec_right */
    arrsize_t amount_ge_pivot
            = partition_vec<vtype>(arr + l_store,
                                   arr + l_store + unpartitioned,
                                   vec_left,
                                   pivot_vec,
                                   min_vec,
                                   max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    unpartitioned -= vtype::numlanes;

    amount_ge_pivot = partition_vec<vtype>(arr + l_store,
                                           arr + l_store + unpartitioned,
                                           vec_right,
                                           pivot_vec,
                                           min_vec,
                                           max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    unpartitioned -= vtype::numlanes;

    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype,
          int num_unroll,
          typename type_t = typename vtype::type_t>
X86_SIMD_SORT_INLINE arrsize_t partition_avx512_unrolled(type_t *arr,
                                                         arrsize_t left,
                                                         arrsize_t right,
                                                         type_t pivot,
                                                         type_t *smallest,
                                                         type_t *biggest)
{
    if constexpr (num_unroll == 0) {
        return partition_avx512<vtype>(
                arr, left, right, pivot, smallest, biggest);
    }

    /* Use regular partition_avx512 for smaller arrays */
    if (right - left < 3 * num_unroll * vtype::numlanes) {
        return partition_avx512<vtype>(
                arr, left, right, pivot, smallest, biggest);
    }

    /* make array length divisible by vtype::numlanes, shortening the array */
    for (int32_t i = ((right - left) % (vtype::numlanes)); i > 0; --i) {
        *smallest = std::min(*smallest, arr[left], comparison_func<vtype>);
        *biggest = std::max(*biggest, arr[left], comparison_func<vtype>);
        if (!comparison_func<vtype>(arr[left], pivot)) {
            std::swap(arr[left], arr[--right]);
        }
        else {
            ++left;
        }
    }

    arrsize_t unpartitioned = right - left - vtype::numlanes;
    arrsize_t l_store = left;

    using reg_t = typename vtype::reg_t;
    reg_t pivot_vec = vtype::set1(pivot);
    reg_t min_vec = vtype::set1(*smallest);
    reg_t max_vec = vtype::set1(*biggest);

    /* Calculate and load more registers to make the rest of the array a
     * multiple of num_unroll. These registers will be partitioned at the very
     * end. */
    int vecsToPartition = ((right - left) / vtype::numlanes) % num_unroll;
    reg_t vec_align[num_unroll];
    for (int i = 0; i < vecsToPartition; i++) {
        vec_align[i] = vtype::loadu(arr + left + i * vtype::numlanes);
    }
    left += vecsToPartition * vtype::numlanes;

    /* We will now have atleast 3*num_unroll registers worth of data to
     * process. Load left and right vtype::numlanes*num_unroll values into
     * registers to make space for in-place parition. The vec_left and
     * vec_right registers are partitioned at the end */
    reg_t vec_left[num_unroll], vec_right[num_unroll];
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        vec_left[ii] = vtype::loadu(arr + left + vtype::numlanes * ii);
        vec_right[ii] = vtype::loadu(
                arr + (right - vtype::numlanes * (num_unroll - ii)));
    }
    /* indices for loading the elements */
    left += num_unroll * vtype::numlanes;
    right -= num_unroll * vtype::numlanes;
    while (right - left != 0) {
        reg_t curr_vec[num_unroll];
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((l_store + unpartitioned + vtype::numlanes) - right
            < left - l_store) {
            right -= num_unroll * vtype::numlanes;
            X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + right + ii * vtype::numlanes);
                /*
                 * error: '_mm_prefetch' needs target feature mmx on clang-cl
                 */
#if !(defined(_MSC_VER) && defined(__clang__))
                _mm_prefetch((char *)(arr + right + ii * vtype::numlanes
                                      - num_unroll * vtype::numlanes),
                             _MM_HINT_T0);
#endif
            }
        }
        else {
            X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + left + ii * vtype::numlanes);
                /*
                 * error: '_mm_prefetch' needs target feature mmx on clang-cl
                 */
#if !(defined(_MSC_VER) && defined(__clang__))
                _mm_prefetch((char *)(arr + left + ii * vtype::numlanes
                                      + num_unroll * vtype::numlanes),
                             _MM_HINT_T0);
#endif
            }
            left += num_unroll * vtype::numlanes;
        }
        /* partition the current vector and save it on both sides of the array
         * */
        X86_SIMD_SORT_UNROLL_LOOP(8)
        for (int ii = 0; ii < num_unroll; ++ii) {
            arrsize_t amount_ge_pivot
                    = partition_vec<vtype>(arr + l_store,
                                           arr + l_store + unpartitioned,
                                           curr_vec[ii],
                                           pivot_vec,
                                           min_vec,
                                           max_vec);
            l_store += (vtype::numlanes - amount_ge_pivot);
            unpartitioned -= vtype::numlanes;
        }
    }

    /* partition and save vec_left[num_unroll] and vec_right[num_unroll] */
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        arrsize_t amount_ge_pivot
                = partition_vec<vtype>(arr + l_store,
                                       arr + l_store + unpartitioned,
                                       vec_left[ii],
                                       pivot_vec,
                                       min_vec,
                                       max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        arrsize_t amount_ge_pivot
                = partition_vec<vtype>(arr + l_store,
                                       arr + l_store + unpartitioned,
                                       vec_right[ii],
                                       pivot_vec,
                                       min_vec,
                                       max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    /* partition and save vec_align[vecsToPartition] */
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < vecsToPartition; ++ii) {
        arrsize_t amount_ge_pivot
                = partition_vec<vtype>(arr + l_store,
                                       arr + l_store + unpartitioned,
                                       vec_align[ii],
                                       pivot_vec,
                                       min_vec,
                                       max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}


template <typename vtype, int maxN>
void sort_n(typename vtype::type_t *arr, int N);

template <typename vtype, typename type_t>
static void qsort_(type_t *arr, arrsize_t left, arrsize_t right, arrsize_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1, comparison_func<vtype>);
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

    type_t pivot = get_pivot_blocks<vtype, type_t>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();

    arrsize_t pivot_index
            = partition_avx512_unrolled<vtype, vtype::partition_unroll_factor>(
                    arr, left, right + 1, pivot, &smallest, &biggest);

    if (pivot != smallest)
        qsort_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest) qsort_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void qselect_(type_t *arr,
                                   arrsize_t pos,
                                   arrsize_t left,
                                   arrsize_t right,
                                   arrsize_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1, comparison_func<vtype>);
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

    arrsize_t pivot_index
            = partition_avx512_unrolled<vtype, vtype::partition_unroll_factor>(
                    arr, left, right + 1, pivot, &smallest, &biggest);

    if ((pivot != smallest) && (pos < pivot_index))
        qselect_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

// Regular quicksort routines:
template <typename T>
X86_SIMD_SORT_INLINE void avx512_qsort(T *arr, arrsize_t arrsize)
{
    if (arrsize > 1) {
        /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
        if constexpr (std::is_floating_point_v<T>) {
            arrsize_t nan_count
                    = replace_nan_with_inf<zmm_vector<T>>(arr, arrsize);
            qsort_<zmm_vector<T>, T>(
                    arr, 0, arrsize - 1, 2 * (arrsize_t)log2(arrsize));
            replace_inf_with_nan(arr, arrsize, nan_count);
        }
        else {
            qsort_<zmm_vector<T>, T>(
                    arr, 0, arrsize - 1, 2 * (arrsize_t)log2(arrsize));
        }
    }
}

template <typename T>
void avx2_qsort(T *arr, int64_t arrsize)
{
    using vtype = xss::avx2::ymm_vector<T>;
    if (arrsize > 1) {
        /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
        if constexpr (std::is_floating_point_v<T>) {
            int64_t nan_count
                    = xss::avx2::replace_nan_with_inf(arr, arrsize);
            qsort_<vtype, T>(
                    arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
            replace_inf_with_nan(arr, arrsize, nan_count);
        }
        else {
            qsort_<vtype, T>(
                    arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        }
    }
}

void avx512_qsort_fp16(uint16_t *arr, int64_t arrsize);

template <typename T>
X86_SIMD_SORT_INLINE void
avx512_qselect(T *arr, arrsize_t k, arrsize_t arrsize, bool hasnan = false)
{
    arrsize_t indx_last_elem = arrsize - 1;
    /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
    if constexpr (std::is_floating_point_v<T>) {
        if (UNLIKELY(hasnan)) {
            indx_last_elem = move_nans_to_end_of_array(arr, arrsize);
        }
    }
    UNUSED(hasnan);
    if (indx_last_elem >= k) {
        qselect_<zmm_vector<T>, T>(
                arr, k, 0, indx_last_elem, 2 * (arrsize_t)log2(indx_last_elem));
    }
}

template <typename T>
void avx2_qselect(T *arr, int64_t k, int64_t arrsize, bool hasnan = false)
{
    int64_t indx_last_elem = arrsize - 1;
    /* std::is_floating_point_v<_Float16> == False, unless c++-23*/
    if constexpr (std::is_floating_point_v<T>) {
        if (UNLIKELY(hasnan)) {
            indx_last_elem = move_nans_to_end_of_array(arr, arrsize);
        }
    }
    UNUSED(hasnan);
    if (indx_last_elem >= k) {
        qselect_<xss::avx2::ymm_vector<T>, T>(
                arr, k, 0, indx_last_elem, 2 * (int64_t)log2(indx_last_elem));
    }
}

void avx512_qselect_fp16(uint16_t *arr,
                         int64_t k,
                         int64_t arrsize,
                         bool hasnan = false);

template <typename T>
X86_SIMD_SORT_INLINE void avx512_partial_qsort(T *arr,
                                               arrsize_t k,
                                               arrsize_t arrsize,
                                               bool hasnan = false)
{
    avx512_qselect<T>(arr, k - 1, arrsize, hasnan);
    avx512_qsort<T>(arr, k - 1);
}

template <typename T>
inline void avx2_partial_qsort(T *arr, int64_t k, int64_t arrsize, bool hasnan = false)
{
    avx2_qselect<T>(arr, k - 1, arrsize, hasnan);
    avx2_qsort<T>(arr, k - 1);
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
