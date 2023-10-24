// SPR specific routines:
#include "avx512fp16-16bit-qsort.hpp"
#include "x86simdsort-internal.h"

namespace xss {
namespace avx512 {
    template <>
    void qsort(_Float16 *arr, size_t size, bool hasnan)
    {
        avx512_qsort(arr, size, hasnan);
    }
    template <>
    void qselect(_Float16 *arr, size_t k, size_t arrsize, bool hasnan)
    {
        avx512_qselect(arr, k, arrsize, hasnan);
    }
    template <>
    void partial_qsort(_Float16 *arr, size_t k, size_t arrsize, bool hasnan)
    {
        avx512_partial_qsort(arr, k, arrsize, hasnan);
    }
} // namespace avx512
} // namespace xss
