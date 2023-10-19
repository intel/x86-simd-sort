/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_ARGSORT_COMMON
#define AVX512_ARGSORT_COMMON

#include "avx512-64bit-common.h"
#include <numeric>
#include <stdio.h>
#include <vector>

/* Workaround for NumPy failed build on macOS x86_64: implicit instantiation of
 * undefined template 'zmm_vector<unsigned long>'*/
#ifdef __APPLE__
using argtype = typename std::conditional<sizeof(arrsize_t) == sizeof(int32_t),
                                          ymm_vector<uint32_t>,
                                          zmm_vector<uint64_t>>::type;
#else
using argtype = typename std::conditional<sizeof(arrsize_t) == sizeof(int32_t),
                                          ymm_vector<arrsize_t>,
                                          zmm_vector<arrsize_t>>::type;
#endif
using argreg_t = typename argtype::reg_t;

/*
 * Parition one ZMM register based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t, typename reg_t>
X86_SIMD_SORT_INLINE int32_t partition_vec(type_t *arg,
                                           arrsize_t left,
                                           arrsize_t right,
                                           const argreg_t arg_vec,
                                           const reg_t curr_vec,
                                           const reg_t pivot_vec,
                                           reg_t *smallest_vec,
                                           reg_t *biggest_vec)
{
    /* which elements are larger than the pivot */
    typename vtype::opmask_t gt_mask = vtype::ge(curr_vec, pivot_vec);
    int32_t amount_gt_pivot = _mm_popcnt_u32((int32_t)gt_mask);
    argtype::mask_compressstoreu(
            arg + left, vtype::knot_opmask(gt_mask), arg_vec);
    argtype::mask_compressstoreu(
            arg + right - amount_gt_pivot, gt_mask, arg_vec);
    *smallest_vec = vtype::min(curr_vec, *smallest_vec);
    *biggest_vec = vtype::max(curr_vec, *biggest_vec);
    return amount_gt_pivot;
}
/*
 * Parition an array based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE arrsize_t partition_avx512(type_t *arr,
                                                arrsize_t *arg,
                                                arrsize_t left,
                                                arrsize_t right,
                                                type_t pivot,
                                                type_t *smallest,
                                                type_t *biggest)
{
    /* make array length divisible by vtype::numlanes , shortening the array */
    for (int32_t i = (right - left) % vtype::numlanes; i > 0; --i) {
        *smallest = std::min(*smallest, arr[arg[left]], comparison_func<vtype>);
        *biggest = std::max(*biggest, arr[arg[left]], comparison_func<vtype>);
        if (!comparison_func<vtype>(arr[arg[left]], pivot)) {
            std::swap(arg[left], arg[--right]);
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
        argreg_t argvec = argtype::loadu(arg + left);
        reg_t vec = vtype::i64gather(arr, arg + left);
        int32_t amount_gt_pivot = partition_vec<vtype>(arg,
                                                       left,
                                                       left + vtype::numlanes,
                                                       argvec,
                                                       vec,
                                                       pivot_vec,
                                                       &min_vec,
                                                       &max_vec);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return left + (vtype::numlanes - amount_gt_pivot);
    }

    // first and last vtype::numlanes values are partitioned at the end
    argreg_t argvec_left = argtype::loadu(arg + left);
    reg_t vec_left = vtype::i64gather(arr, arg + left);
    argreg_t argvec_right = argtype::loadu(arg + (right - vtype::numlanes));
    reg_t vec_right = vtype::i64gather(arr, arg + (right - vtype::numlanes));
    // store points of the vectors
    arrsize_t r_store = right - vtype::numlanes;
    arrsize_t l_store = left;
    // indices for loading the elements
    left += vtype::numlanes;
    right -= vtype::numlanes;
    while (right - left != 0) {
        argreg_t arg_vec;
        reg_t curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + vtype::numlanes) - right < left - l_store) {
            right -= vtype::numlanes;
            arg_vec = argtype::loadu(arg + right);
            curr_vec = vtype::i64gather(arr, arg + right);
        }
        else {
            arg_vec = argtype::loadu(arg + left);
            curr_vec = vtype::i64gather(arr, arg + left);
            left += vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        int32_t amount_gt_pivot
                = partition_vec<vtype>(arg,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       arg_vec,
                                       curr_vec,
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        ;
        r_store -= amount_gt_pivot;
        l_store += (vtype::numlanes - amount_gt_pivot);
    }

    /* partition and save vec_left and vec_right */
    int32_t amount_gt_pivot = partition_vec<vtype>(arg,
                                                   l_store,
                                                   r_store + vtype::numlanes,
                                                   argvec_left,
                                                   vec_left,
                                                   pivot_vec,
                                                   &min_vec,
                                                   &max_vec);
    l_store += (vtype::numlanes - amount_gt_pivot);
    amount_gt_pivot = partition_vec<vtype>(arg,
                                           l_store,
                                           l_store + vtype::numlanes,
                                           argvec_right,
                                           vec_right,
                                           pivot_vec,
                                           &min_vec,
                                           &max_vec);
    l_store += (vtype::numlanes - amount_gt_pivot);
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype,
          int num_unroll,
          typename type_t = typename vtype::type_t>
X86_SIMD_SORT_INLINE arrsize_t partition_avx512_unrolled(type_t *arr,
                                                         arrsize_t *arg,
                                                         arrsize_t left,
                                                         arrsize_t right,
                                                         type_t pivot,
                                                         type_t *smallest,
                                                         type_t *biggest)
{
    if (right - left <= 8 * num_unroll * vtype::numlanes) {
        return partition_avx512<vtype>(
                arr, arg, left, right, pivot, smallest, biggest);
    }
    /* make array length divisible by vtype::numlanes , shortening the array */
    for (int32_t i = ((right - left) % (num_unroll * vtype::numlanes)); i > 0;
         --i) {
        *smallest = std::min(*smallest, arr[arg[left]], comparison_func<vtype>);
        *biggest = std::max(*biggest, arr[arg[left]], comparison_func<vtype>);
        if (!comparison_func<vtype>(arr[arg[left]], pivot)) {
            std::swap(arg[left], arg[--right]);
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

    // first and last vtype::numlanes values are partitioned at the end
    reg_t vec_left[num_unroll], vec_right[num_unroll];
    argreg_t argvec_left[num_unroll], argvec_right[num_unroll];
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        argvec_left[ii] = argtype::loadu(arg + left + vtype::numlanes * ii);
        vec_left[ii] = vtype::i64gather(arr, arg + left + vtype::numlanes * ii);
        argvec_right[ii] = argtype::loadu(
                arg + (right - vtype::numlanes * (num_unroll - ii)));
        vec_right[ii] = vtype::i64gather(
                arr, arg + (right - vtype::numlanes * (num_unroll - ii)));
    }
    // store points of the vectors
    arrsize_t r_store = right - vtype::numlanes;
    arrsize_t l_store = left;
    // indices for loading the elements
    left += num_unroll * vtype::numlanes;
    right -= num_unroll * vtype::numlanes;
    while (right - left != 0) {
        argreg_t arg_vec[num_unroll];
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
                arg_vec[ii]
                        = argtype::loadu(arg + right + ii * vtype::numlanes);
                curr_vec[ii] = vtype::i64gather(
                        arr, arg + right + ii * vtype::numlanes);
            }
        }
        else {
            X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                arg_vec[ii] = argtype::loadu(arg + left + ii * vtype::numlanes);
                curr_vec[ii] = vtype::i64gather(
                        arr, arg + left + ii * vtype::numlanes);
            }
            left += num_unroll * vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        X86_SIMD_SORT_UNROLL_LOOP(8)
        for (int ii = 0; ii < num_unroll; ++ii) {
            int32_t amount_gt_pivot
                    = partition_vec<vtype>(arg,
                                           l_store,
                                           r_store + vtype::numlanes,
                                           arg_vec[ii],
                                           curr_vec[ii],
                                           pivot_vec,
                                           &min_vec,
                                           &max_vec);
            l_store += (vtype::numlanes - amount_gt_pivot);
            r_store -= amount_gt_pivot;
        }
    }

    /* partition and save vec_left and vec_right */
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        int32_t amount_gt_pivot
                = partition_vec<vtype>(arg,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       argvec_left[ii],
                                       vec_left[ii],
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        l_store += (vtype::numlanes - amount_gt_pivot);
        r_store -= amount_gt_pivot;
    }
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        int32_t amount_gt_pivot
                = partition_vec<vtype>(arg,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       argvec_right[ii],
                                       vec_right[ii],
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        l_store += (vtype::numlanes - amount_gt_pivot);
        r_store -= amount_gt_pivot;
    }
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}
#endif // AVX512_ARGSORT_COMMON
