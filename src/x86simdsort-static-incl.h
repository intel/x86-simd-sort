#ifndef X86_SIMD_SORT_STATIC_METHODS
#define X86_SIMD_SORT_STATIC_METHODS
#include <vector>
#include <stdlib.h>
// Declare all methods:
namespace x86simdsortStatic {
template <typename T>
X86_SIMD_SORT_FINLINE void qsort(T *arr, size_t size, bool hasnan = true);

template <typename T>
X86_SIMD_SORT_FINLINE void
qselect(T *arr, size_t k, size_t size, bool hasnan = true);

template <typename T>
X86_SIMD_SORT_FINLINE void
partial_qsort(T *arr, size_t k, size_t size, bool hasnan = true);

template <typename T>
X86_SIMD_SORT_FINLINE std::vector<size_t>
argsort(T *arr, size_t size, bool hasnan = true);

template <typename T>
std::vector<size_t> X86_SIMD_SORT_FINLINE
argselect(T *arr, size_t k, size_t size, bool hasnan = true);

template <typename T1, typename T2>
X86_SIMD_SORT_FINLINE void
keyvalue_qsort(T1 *key, T2 *val, size_t size, bool hasnan = true);
} // namespace x86simdsortStatic

#define XSS_METHODS(ISA) \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::qsort( \
            T *arr, size_t size, bool hasnan) \
    { \
        ISA##_qsort(arr, size, hasnan); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::qselect( \
            T *arr, size_t k, size_t size, bool hasnan) \
    { \
        ISA##_qselect(arr, k, size, hasnan); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::partial_qsort( \
            T *arr, size_t k, size_t size, bool hasnan) \
    { \
        ISA##_partial_qsort(arr, k, size, hasnan); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE std::vector<size_t> x86simdsortStatic::argsort( \
            T *arr, size_t size, bool hasnan) \
    { \
        return ISA##_argsort(arr, size, hasnan); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE std::vector<size_t> x86simdsortStatic::argselect( \
            T *arr, size_t k, size_t size, bool hasnan) \
    { \
        return ISA##_argselect(arr, k, size, hasnan); \
    }

/*
 * qsort, qselect, partial, argsort key-value sort template functions.
 */
#include "xss-common-qsort.h"
#include "xss-common-argsort.h"

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
/* 32-bit and 64-bit dtypes vector definitions on SKX */
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"
#include "avx512-64bit-argsort.hpp"
#include "avx512-64bit-keyvaluesort.hpp"

/* 16-bit dtypes vector definitions on ICL */
#if defined(__AVX512BW__) && defined(__AVX512VBMI2__)
#include "avx512-16bit-qsort.hpp"
/* _Float16 vector definition on SPR*/
#if defined(__FLT16_MAX__) && defined(__AVX512BW__) && defined(__AVX512FP16__)
#include "avx512fp16-16bit-qsort.hpp"
#endif // __FLT16_MAX__
#endif // __AVX512VBMI2__

XSS_METHODS(avx512)

// key-value currently only on avx512
template <typename T1, typename T2>
X86_SIMD_SORT_FINLINE void
x86simdsortStatic::keyvalue_qsort(T1 *key, T2 *val, size_t size, bool hasnan)
{
    avx512_qsort_kv(key, val, size, hasnan);
}

#elif defined(__AVX2__) && !defined(__AVX512F__)
/* 32-bit and 64-bit dtypes vector definitions on AVX2 */
#include "avx2-32bit-half.hpp"
#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
XSS_METHODS(avx2)

#else
#error "x86simdsortStatic methods needs to be compiled with avx512/avx2 specific flags"
#endif // (__AVX512VL__ && __AVX512DQ__) || AVX2

#endif // X86_SIMD_SORT_STATIC_METHODS
