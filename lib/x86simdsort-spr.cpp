// SPR specific routines:
#include "x86simdsort-static-incl.h"
#include "x86simdsort-internal.h"

namespace xss {
namespace avx512 {
    template <>
    void qsort(_Float16 *arr, size_t size, bool hasnan, bool descending)
    {
        if (descending) { x86simdsortStatic::qsort(arr, size, hasnan, true); }
        else {
            x86simdsortStatic::qsort(arr, size, hasnan, false);
        }
    }
    template <>
    void qselect(_Float16 *arr,
                 size_t k,
                 size_t arrsize,
                 bool hasnan,
                 bool descending)
    {
        if (descending) {
            x86simdsortStatic::qselect(arr, k, arrsize, hasnan, true);
        }
        else {
            x86simdsortStatic::qselect(arr, k, arrsize, hasnan, false);
        }
    }
    template <>
    void partial_qsort(_Float16 *arr,
                       size_t k,
                       size_t arrsize,
                       bool hasnan,
                       bool descending)
    {
        if (descending) {
            x86simdsortStatic::partial_qsort(arr, k, arrsize, hasnan, true);
        }
        else {
            x86simdsortStatic::partial_qsort(arr, k, arrsize, hasnan, false);
        }
    }
} // namespace avx512
} // namespace xss
