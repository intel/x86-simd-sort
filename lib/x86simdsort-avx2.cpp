// AVX2 specific routines:
#include "avx2-32bit-qsort.hpp"
#include "x86simdsort-internal.h"

#define DEFINE_ALL_METHODS(type) \
    template <> \
    void qsort(type *arr, size_t arrsize, bool hasnan) \
    { \
        avx2_qsort(arr, arrsize, hasnan); \
    } \
    template <> \
    void qselect(type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        avx2_qselect(arr, k, arrsize, hasnan); \
    } \
    template <> \
    void partial_qsort(type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        avx2_partial_qsort(arr, k, arrsize, hasnan); \
    }

namespace xss {
namespace avx2 {
    DEFINE_ALL_METHODS(uint32_t)
    DEFINE_ALL_METHODS(int32_t)
    DEFINE_ALL_METHODS(float)
} // namespace avx2
} // namespace xss
