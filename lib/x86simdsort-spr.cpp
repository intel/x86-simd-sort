// SPR specific routines:
#include "avx512fp16-16bit-qsort.hpp"
#include "x86simdsort-internal.h"

namespace xss {
namespace avx512 {
    template <>
    void qsort(_Float16 *arr, size_t size, bool hasnan, bool descending)
    {
        if (descending) { avx512_qsort<true>(arr, size, hasnan); }
        else {
            avx512_qsort<false>(arr, size, hasnan);
        }
    }
    template <>
    void qselect(_Float16 *arr,
                 size_t k,
                 size_t arrsize,
                 bool hasnan,
                 bool descending)
    {
        if (descending) { avx512_qselect<true>(arr, k, arrsize, hasnan); }
        else {
            avx512_qselect<false>(arr, k, arrsize, hasnan);
        }
    }
    template <>
    void partial_qsort(_Float16 *arr,
                       size_t k,
                       size_t arrsize,
                       bool hasnan,
                       bool descending)
    {
        if (descending) { avx512_partial_qsort<true>(arr, k, arrsize, hasnan); }
        else {
            avx512_partial_qsort<false>(arr, k, arrsize, hasnan);
        }
    }
} // namespace avx512
} // namespace xss
