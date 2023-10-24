/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_ARGSORT_64BIT
#define AVX512_ARGSORT_64BIT

#include "xss-common-qsort.h"
#include "avx512-64bit-common.h"
#include "xss-network-keyvaluesort.hpp"
#include <numeric>

template <typename T>
X86_SIMD_SORT_INLINE void std_argselect_withnan(
        T *arr, arrsize_t *arg, arrsize_t k, arrsize_t left, arrsize_t right)
{
    std::nth_element(arg + left,
                     arg + k,
                     arg + right,
                     [arr](arrsize_t a, arrsize_t b) -> bool {
                         if ((!std::isnan(arr[a])) && (!std::isnan(arr[b]))) {
                             return arr[a] < arr[b];
                         }
                         else if (std::isnan(arr[a])) {
                             return false;
                         }
                         else {
                             return true;
                         }
                     });
}

/* argsort using std::sort */
template <typename T>
X86_SIMD_SORT_INLINE void
std_argsort_withnan(T *arr, arrsize_t *arg, arrsize_t left, arrsize_t right)
{
    std::sort(arg + left,
              arg + right,
              [arr](arrsize_t left, arrsize_t right) -> bool {
                  if ((!std::isnan(arr[left])) && (!std::isnan(arr[right]))) {
                      return arr[left] < arr[right];
                  }
                  else if (std::isnan(arr[left])) {
                      return false;
                  }
                  else {
                      return true;
                  }
              });
}

/* argsort using std::sort */
template <typename T>
X86_SIMD_SORT_INLINE void
std_argsort(T *arr, arrsize_t *arg, arrsize_t left, arrsize_t right)
{
    std::sort(arg + left,
              arg + right,
              [arr](arrsize_t left, arrsize_t right) -> bool {
                  // sort indices according to corresponding array element
                  return arr[left] < arr[right];
              });
}

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

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void
argsort_8_64bit(type_t *arr, arrsize_t *arg, int32_t N)
{
    using reg_t = typename vtype::reg_t;
    typename vtype::opmask_t load_mask = (0x01 << N) - 0x01;
    argreg_t argzmm = argtype::maskz_loadu(load_mask, arg);
    reg_t arrzmm = vtype::template mask_i64gather<sizeof(type_t)>(
            vtype::zmm_max(), load_mask, argzmm, arr);
    arrzmm = sort_zmm_64bit<vtype, argtype>(arrzmm, argzmm);
    argtype::mask_storeu(arg, load_mask, argzmm);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void
argsort_16_64bit(type_t *arr, arrsize_t *arg, int32_t N)
{
    if (N <= 8) {
        argsort_8_64bit<vtype>(arr, arg, N);
        return;
    }
    using reg_t = typename vtype::reg_t;
    typename vtype::opmask_t load_mask = (0x01 << (N - 8)) - 0x01;
    argreg_t argzmm1 = argtype::loadu(arg);
    argreg_t argzmm2 = argtype::maskz_loadu(load_mask, arg + 8);
    reg_t arrzmm1 = vtype::i64gather(arr, arg);
    reg_t arrzmm2 = vtype::template mask_i64gather<sizeof(type_t)>(
            vtype::zmm_max(), load_mask, argzmm2, arr);
    arrzmm1 = sort_zmm_64bit<vtype, argtype>(arrzmm1, argzmm1);
    arrzmm2 = sort_zmm_64bit<vtype, argtype>(arrzmm2, argzmm2);
    bitonic_merge_two_zmm_64bit<vtype, argtype>(
            arrzmm1, arrzmm2, argzmm1, argzmm2);
    argtype::storeu(arg, argzmm1);
    argtype::mask_storeu(arg + 8, load_mask, argzmm2);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void
argsort_32_64bit(type_t *arr, arrsize_t *arg, int32_t N)
{
    if (N <= 16) {
        argsort_16_64bit<vtype>(arr, arg, N);
        return;
    }
    using reg_t = typename vtype::reg_t;
    using opmask_t = typename vtype::opmask_t;
    reg_t arrzmm[4];
    argreg_t argzmm[4];

    X86_SIMD_SORT_UNROLL_LOOP(2)
    for (int ii = 0; ii < 2; ++ii) {
        argzmm[ii] = argtype::loadu(arg + 8 * ii);
        arrzmm[ii] = vtype::i64gather(arr, arg + 8 * ii);
        arrzmm[ii] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii], argzmm[ii]);
    }

    uint64_t combined_mask = (0x1ull << (N - 16)) - 0x1ull;
    opmask_t load_mask[2] = {0xFF, 0xFF};
    X86_SIMD_SORT_UNROLL_LOOP(2)
    for (int ii = 0; ii < 2; ++ii) {
        load_mask[ii] = (combined_mask >> (ii * 8)) & 0xFF;
        argzmm[ii + 2] = argtype::maskz_loadu(load_mask[ii], arg + 16 + 8 * ii);
        arrzmm[ii + 2] = vtype::template mask_i64gather<sizeof(type_t)>(
                vtype::zmm_max(), load_mask[ii], argzmm[ii + 2], arr);
        arrzmm[ii + 2] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii + 2],
                                                        argzmm[ii + 2]);
    }

    bitonic_merge_two_zmm_64bit<vtype, argtype>(
            arrzmm[0], arrzmm[1], argzmm[0], argzmm[1]);
    bitonic_merge_two_zmm_64bit<vtype, argtype>(
            arrzmm[2], arrzmm[3], argzmm[2], argzmm[3]);
    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm, argzmm);

    argtype::storeu(arg, argzmm[0]);
    argtype::storeu(arg + 8, argzmm[1]);
    argtype::mask_storeu(arg + 16, load_mask[0], argzmm[2]);
    argtype::mask_storeu(arg + 24, load_mask[1], argzmm[3]);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void
argsort_64_64bit(type_t *arr, arrsize_t *arg, int32_t N)
{
    if (N <= 32) {
        argsort_32_64bit<vtype>(arr, arg, N);
        return;
    }
    using reg_t = typename vtype::reg_t;
    using opmask_t = typename vtype::opmask_t;
    reg_t arrzmm[8];
    argreg_t argzmm[8];

    X86_SIMD_SORT_UNROLL_LOOP(4)
    for (int ii = 0; ii < 4; ++ii) {
        argzmm[ii] = argtype::loadu(arg + 8 * ii);
        arrzmm[ii] = vtype::i64gather(arr, arg + 8 * ii);
        arrzmm[ii] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii], argzmm[ii]);
    }

    opmask_t load_mask[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    uint64_t combined_mask = (0x1ull << (N - 32)) - 0x1ull;
    X86_SIMD_SORT_UNROLL_LOOP(4)
    for (int ii = 0; ii < 4; ++ii) {
        load_mask[ii] = (combined_mask >> (ii * 8)) & 0xFF;
        argzmm[ii + 4] = argtype::maskz_loadu(load_mask[ii], arg + 32 + 8 * ii);
        arrzmm[ii + 4] = vtype::template mask_i64gather<sizeof(type_t)>(
                vtype::zmm_max(), load_mask[ii], argzmm[ii + 4], arr);
        arrzmm[ii + 4] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii + 4],
                                                        argzmm[ii + 4]);
    }

    X86_SIMD_SORT_UNROLL_LOOP(4)
    for (int ii = 0; ii < 8; ii = ii + 2) {
        bitonic_merge_two_zmm_64bit<vtype, argtype>(
                arrzmm[ii], arrzmm[ii + 1], argzmm[ii], argzmm[ii + 1]);
    }
    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm, argzmm);
    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm + 4, argzmm + 4);
    bitonic_merge_eight_zmm_64bit<vtype, argtype>(arrzmm, argzmm);

    X86_SIMD_SORT_UNROLL_LOOP(4)
    for (int ii = 0; ii < 4; ++ii) {
        argtype::storeu(arg + 8 * ii, argzmm[ii]);
    }
    X86_SIMD_SORT_UNROLL_LOOP(4)
    for (int ii = 0; ii < 4; ++ii) {
        argtype::mask_storeu(arg + 32 + 8 * ii, load_mask[ii], argzmm[ii + 4]);
    }
}

/* arsort 128 doesn't seem to make much of a difference to perf*/
//template <typename vtype, typename type_t>
//X86_SIMD_SORT_INLINE void
//argsort_128_64bit(type_t *arr, arrsize_t *arg, int32_t N)
//{
//    if (N <= 64) {
//        argsort_64_64bit<vtype>(arr, arg, N);
//        return;
//    }
//    using reg_t = typename vtype::reg_t;
//    using opmask_t = typename vtype::opmask_t;
//    reg_t arrzmm[16];
//    argreg_t argzmm[16];
//
//X86_SIMD_SORT_UNROLL_LOOP(8)
//    for (int ii = 0; ii < 8; ++ii) {
//        argzmm[ii] = argtype::loadu(arg + 8*ii);
//        arrzmm[ii] = vtype::i64gather(argzmm[ii], arr);
//        arrzmm[ii] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii], argzmm[ii]);
//    }
//
//    opmask_t load_mask[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
//    if (N != 128) {
//    uarrsize_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
//X86_SIMD_SORT_UNROLL_LOOP(8)
//        for (int ii = 0; ii < 8; ++ii) {
//            load_mask[ii] = (combined_mask >> (ii*8)) & 0xFF;
//        }
//    }
//X86_SIMD_SORT_UNROLL_LOOP(8)
//    for (int ii = 0; ii < 8; ++ii) {
//        argzmm[ii+8] = argtype::maskz_loadu(load_mask[ii], arg + 64 + 8*ii);
//        arrzmm[ii+8] = vtype::template mask_i64gather<sizeof(type_t)>(vtype::zmm_max(), load_mask[ii], argzmm[ii+8], arr);
//        arrzmm[ii+8] = sort_zmm_64bit<vtype, argtype>(arrzmm[ii+8], argzmm[ii+8]);
//    }
//
//X86_SIMD_SORT_UNROLL_LOOP(8)
//    for (int ii = 0; ii < 16; ii = ii + 2) {
//        bitonic_merge_two_zmm_64bit<vtype, argtype>(arrzmm[ii], arrzmm[ii + 1], argzmm[ii], argzmm[ii + 1]);
//    }
//    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm, argzmm);
//    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm + 4, argzmm + 4);
//    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm + 8, argzmm + 8);
//    bitonic_merge_four_zmm_64bit<vtype, argtype>(arrzmm + 12, argzmm + 12);
//    bitonic_merge_eight_zmm_64bit<vtype, argtype>(arrzmm, argzmm);
//    bitonic_merge_eight_zmm_64bit<vtype, argtype>(arrzmm+8, argzmm+8);
//    bitonic_merge_sixteen_zmm_64bit<vtype, argtype>(arrzmm, argzmm);
//
//X86_SIMD_SORT_UNROLL_LOOP(8)
//    for (int ii = 0; ii < 8; ++ii) {
//        argtype::storeu(arg + 8*ii, argzmm[ii]);
//    }
//X86_SIMD_SORT_UNROLL_LOOP(8)
//    for (int ii = 0; ii < 8; ++ii) {
//        argtype::mask_storeu(arg + 64 + 8*ii, load_mask[ii], argzmm[ii + 8]);
//    }
//}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_64bit(type_t *arr,
                                            arrsize_t *arg,
                                            const arrsize_t left,
                                            const arrsize_t right)
{
    if (right - left >= vtype::numlanes) {
        // median of 8
        arrsize_t size = (right - left) / 8;
        using reg_t = typename vtype::reg_t;
        reg_t rand_vec = vtype::set(arr[arg[left + size]],
                                    arr[arg[left + 2 * size]],
                                    arr[arg[left + 3 * size]],
                                    arr[arg[left + 4 * size]],
                                    arr[arg[left + 5 * size]],
                                    arr[arg[left + 6 * size]],
                                    arr[arg[left + 7 * size]],
                                    arr[arg[left + 8 * size]]);
        // pivot will never be a nan, since there are no nan's!
        reg_t sort = sort_zmm_64bit<vtype>(rand_vec);
        return ((type_t *)&sort)[4];
    }
    else {
        return arr[arg[right]];
    }
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void argsort_64bit_(type_t *arr,
                                         arrsize_t *arg,
                                         arrsize_t left,
                                         arrsize_t right,
                                         arrsize_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std_argsort(arr, arg, left, right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 64
     */
    if (right + 1 - left <= 64) {
        argsort_64_64bit<vtype>(arr, arg + left, (int32_t)(right + 1 - left));
        return;
    }
    type_t pivot = get_pivot_64bit<vtype>(arr, arg, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    arrsize_t pivot_index = partition_avx512_unrolled<vtype, 4>(
            arr, arg, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        argsort_64bit_<vtype>(arr, arg, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        argsort_64bit_<vtype>(arr, arg, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void argselect_64bit_(type_t *arr,
                                           arrsize_t *arg,
                                           arrsize_t pos,
                                           arrsize_t left,
                                           arrsize_t right,
                                           arrsize_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std_argsort(arr, arg, left, right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 64
     */
    if (right + 1 - left <= 64) {
        argsort_64_64bit<vtype>(arr, arg + left, (int32_t)(right + 1 - left));
        return;
    }
    type_t pivot = get_pivot_64bit<vtype>(arr, arg, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    arrsize_t pivot_index = partition_avx512_unrolled<vtype, 4>(
            arr, arg, left, right + 1, pivot, &smallest, &biggest);
    if ((pivot != smallest) && (pos < pivot_index))
        argselect_64bit_<vtype>(
                arr, arg, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        argselect_64bit_<vtype>(
                arr, arg, pos, pivot_index, right, max_iters - 1);
}

/* argsort methods for 32-bit and 64-bit dtypes */
template <typename T>
X86_SIMD_SORT_INLINE void
avx512_argsort(T *arr, arrsize_t *arg, arrsize_t arrsize, bool hasnan = false)
{
    using vectype = typename std::conditional<sizeof(T) == sizeof(int32_t),
                                              ymm_vector<T>,
                                              zmm_vector<T>>::type;
    if (arrsize > 1) {
        if constexpr (std::is_floating_point_v<T>) {
            if ((hasnan) && (array_has_nan<vectype>(arr, arrsize))) {
                std_argsort_withnan(arr, arg, 0, arrsize);
                return;
            }
        }
        UNUSED(hasnan);
        argsort_64bit_<vectype>(
                arr, arg, 0, arrsize - 1, 2 * (arrsize_t)log2(arrsize));
    }
}

template <typename T>
X86_SIMD_SORT_INLINE std::vector<arrsize_t> avx512_argsort(T *arr,
                                                           arrsize_t arrsize,
                                                           bool hasnan = false)
{
    std::vector<arrsize_t> indices(arrsize);
    std::iota(indices.begin(), indices.end(), 0);
    avx512_argsort<T>(arr, indices.data(), arrsize, hasnan);
    return indices;
}

/* argselect methods for 32-bit and 64-bit dtypes */
template <typename T>
X86_SIMD_SORT_INLINE void
avx512_argselect(T *arr, arrsize_t *arg, arrsize_t k, arrsize_t arrsize, bool hasnan = false)
{
    using vectype = typename std::conditional<sizeof(T) == sizeof(int32_t),
                                              ymm_vector<T>,
                                              zmm_vector<T>>::type;

    if (arrsize > 1) {
        if constexpr (std::is_floating_point_v<T>) {
            if ((hasnan) && (array_has_nan<vectype>(arr, arrsize))) {
                std_argselect_withnan(arr, arg, k, 0, arrsize);
                return;
            }
        }
        UNUSED(hasnan);
        argselect_64bit_<vectype>(
                arr, arg, k, 0, arrsize - 1, 2 * (arrsize_t)log2(arrsize));
    }
}

template <typename T>
X86_SIMD_SORT_INLINE std::vector<arrsize_t>
avx512_argselect(T *arr, arrsize_t k, arrsize_t arrsize, bool hasnan = false)
{
    std::vector<arrsize_t> indices(arrsize);
    std::iota(indices.begin(), indices.end(), 0);
    avx512_argselect<T>(arr, indices.data(), k, arrsize, hasnan);
    return indices;
}

/* To maintain compatibility with NumPy build */
template <typename T>
X86_SIMD_SORT_INLINE void
avx512_argselect(T *arr, int64_t *arg, arrsize_t k, arrsize_t arrsize)
{
    avx512_argselect(arr, reinterpret_cast<arrsize_t *>(arg), k, arrsize);
}

template <typename T>
X86_SIMD_SORT_INLINE void
avx512_argsort(T *arr, int64_t *arg, arrsize_t arrsize)
{
    avx512_argsort(arr, reinterpret_cast<arrsize_t *>(arg), arrsize);
}

#endif // AVX512_ARGSORT_64BIT
