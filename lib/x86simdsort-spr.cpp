// SPR specific routines:
#include "avx512fp16-16bit-qsort.hpp"
#include "x86simdsort-internal.h"

using x86simdsort::sort_order;

namespace xss {
namespace avx512 {
    template <>
    void qsort(_Float16 *arr, size_t size, bool hasnan, sort_order order)
    {
        avx512_qsort(arr, size, hasnan, order);
    }
    template <>
    void qselect(_Float16 *arr,
                 size_t k,
                 size_t arrsize,
                 bool hasnan,
                 sort_order order)
    {
        avx512_qselect(arr, k, arrsize, hasnan, order);
    }
    template <>
    void partial_qsort(_Float16 *arr,
                       size_t k,
                       size_t arrsize,
                       bool hasnan,
                       sort_order order)
    {
        avx512_partial_qsort(arr, k, arrsize, hasnan, order);
    }
} // namespace avx512
} // namespace xss
