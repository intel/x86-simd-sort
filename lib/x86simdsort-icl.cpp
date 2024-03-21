// ICL specific routines:
#include "avx512-16bit-qsort.hpp"
#include "x86simdsort-internal.h"

using x86simdsort::sort_order;

namespace xss {
namespace avx512 {
    template <>
    void qsort(uint16_t *arr, size_t size, bool hasnan, sort_order order)
    {
        avx512_qsort(arr, size, hasnan, order);
    }
    template <>
    void qselect(uint16_t *arr,
                 size_t k,
                 size_t arrsize,
                 bool hasnan,
                 sort_order order)
    {
        avx512_qselect(arr, k, arrsize, hasnan, order);
    }
    template <>
    void partial_qsort(uint16_t *arr,
                       size_t k,
                       size_t arrsize,
                       bool hasnan,
                       sort_order order)
    {
        avx512_partial_qsort(arr, k, arrsize, hasnan, order);
    }
    template <>
    void qsort(int16_t *arr, size_t size, bool hasnan, sort_order order)
    {
        avx512_qsort(arr, size, hasnan, order);
    }
    template <>
    void qselect(int16_t *arr,
                 size_t k,
                 size_t arrsize,
                 bool hasnan,
                 sort_order order)
    {
        avx512_qselect(arr, k, arrsize, hasnan, order);
    }
    template <>
    void partial_qsort(int16_t *arr,
                       size_t k,
                       size_t arrsize,
                       bool hasnan,
                       sort_order order)
    {
        avx512_partial_qsort(arr, k, arrsize, hasnan, order);
    }
} // namespace avx512
} // namespace xss
