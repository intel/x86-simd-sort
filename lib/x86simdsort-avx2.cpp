// AVX2 specific routines:
#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
#include "avx2-32bit-half.hpp"
#include "xss-common-argsort.h"
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
    }\
    template <> \
    std::vector<size_t> argsort(type *arr, size_t arrsize, bool hasnan) \
    { \
        return avx2_argsort(arr, arrsize, hasnan); \
    } \
    template <> \
    std::vector<size_t> argselect( \
            type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        return avx2_argselect(arr, k, arrsize, hasnan); \
    }

namespace xss {
namespace avx2 {
    DEFINE_ALL_METHODS(uint32_t)
    DEFINE_ALL_METHODS(int32_t)
    DEFINE_ALL_METHODS(float)
    DEFINE_ALL_METHODS(uint64_t)
    DEFINE_ALL_METHODS(int64_t)
    DEFINE_ALL_METHODS(double)
} // namespace avx2
} // namespace xss
