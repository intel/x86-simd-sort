// SKX specific routines:
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-keyvaluesort.hpp"
#include "avx512-64bit-argsort.hpp"
#include "avx512-64bit-qsort.hpp"
#include "x86simdsort-internal.h"

#define DEFINE_ALL_METHODS(type) \
    template <> \
    void qsort(type *arr, size_t arrsize, bool hasnan) \
    { \
        avx512_qsort(arr, arrsize, hasnan); \
    } \
    template <> \
    void qselect(type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        avx512_qselect(arr, k, arrsize, hasnan); \
    } \
    template <> \
    void partial_qsort(type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        avx512_partial_qsort(arr, k, arrsize, hasnan); \
    } \
    template <> \
    std::vector<size_t> argsort(type *arr, size_t arrsize, bool hasnan) \
    { \
        return avx512_argsort(arr, arrsize, hasnan); \
    } \
    template <> \
    std::vector<size_t> argselect( \
            type *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        return avx512_argselect(arr, k, arrsize, hasnan); \
    }

#define DEFINE_KEYVALUE_METHODS(type1, type2) \
    template <> \
    void keyvalue_qsort(type1 *key, type2* val, size_t arrsize, bool hasnan) \
    { \
        avx512_qsort_kv(key, val, arrsize, hasnan); \
    } \


namespace xss {
namespace avx512 {
    DEFINE_ALL_METHODS(uint32_t)
    DEFINE_ALL_METHODS(int32_t)
    DEFINE_ALL_METHODS(float)
    DEFINE_ALL_METHODS(uint64_t)
    DEFINE_ALL_METHODS(int64_t)
    DEFINE_ALL_METHODS(double)
    DEFINE_KEYVALUE_METHODS(double, uint64_t)
    DEFINE_KEYVALUE_METHODS(double, int64_t)
    DEFINE_KEYVALUE_METHODS(double, double)
    DEFINE_KEYVALUE_METHODS(uint64_t, uint64_t)
    DEFINE_KEYVALUE_METHODS(uint64_t, int64_t)
    DEFINE_KEYVALUE_METHODS(uint64_t, double)
    DEFINE_KEYVALUE_METHODS(int64_t, uint64_t)
    DEFINE_KEYVALUE_METHODS(int64_t, int64_t)
    DEFINE_KEYVALUE_METHODS(int64_t, double)
} // namespace avx512
} // namespace xss
