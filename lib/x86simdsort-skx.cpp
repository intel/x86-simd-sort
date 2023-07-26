// SKX specific routines:
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-argsort.hpp"
#include "avx512-64bit-qsort.hpp"
#include "x86simdsort-internal.h"

#define DEFINE_ALL_METHODS(type) \
    template <> \
    void qsort(type *arr, int64_t arrsize) \
    { \
        avx512_qsort(arr, arrsize); \
    } \
    template <> \
    void qselect(type *arr, int64_t k, int64_t arrsize, bool hasnan) \
    { \
        avx512_qselect(arr, k, arrsize, hasnan); \
    } \
    template <> \
    void partial_qsort(type *arr, int64_t k, int64_t arrsize, bool hasnan) \
    { \
        avx512_partial_qsort(arr, k, arrsize, hasnan); \
    } \
    template <> \
    std::vector<int64_t> argsort(type *arr, int64_t arrsize) \
    { \
        return avx512_argsort(arr, arrsize); \
    } \
    template <> \
    std::vector<int64_t> argselect(type *arr, int64_t k, int64_t arrsize) \
    { \
        return avx512_argselect(arr, k, arrsize); \
    }

namespace xss {
namespace avx512 {
    DEFINE_ALL_METHODS(uint32_t)
    DEFINE_ALL_METHODS(int32_t)
    DEFINE_ALL_METHODS(float)
    DEFINE_ALL_METHODS(uint64_t)
    DEFINE_ALL_METHODS(int64_t)
    DEFINE_ALL_METHODS(double)
} // namespace avx512
} // namespace xss
