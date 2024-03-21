#ifndef XSS_INTERNAL_METHODS
#define XSS_INTERNAL_METHODS
#include "x86simdsort.h"
#include <stdint.h>
#include <vector>

using x86simdsort::sort_order;

namespace xss {
namespace avx512 {
    // quicksort
    template <typename T>
    XSS_HIDE_SYMBOL void qsort(T *arr,
                               size_t arrsize,
                               bool hasnan = false,
                               sort_order order = sort_order::sort_ascending);
    // key-value quicksort
    template <typename T1, typename T2>
    XSS_EXPORT_SYMBOL void
    keyvalue_qsort(T1 *key, T2 *val, size_t arrsize, bool hasnan = false);
    // quickselect
    template <typename T>
    XSS_HIDE_SYMBOL void qselect(T *arr,
                                 size_t k,
                                 size_t arrsize,
                                 bool hasnan = false,
                                 sort_order order = sort_order::sort_ascending);
    // partial sort
    template <typename T>
    XSS_HIDE_SYMBOL void partial_qsort(T *arr,
                                       size_t k,
                                       size_t arrsize,
                                       bool hasnan = false,
                                       sort_order order
                                       = sort_order::sort_ascending);
    // argsort
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argsort(T *arr, size_t arrsize, bool hasnan = false);
    // argselect
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
} // namespace avx512
namespace avx2 {
    // quicksort
    template <typename T>
    XSS_HIDE_SYMBOL void qsort(T *arr,
                               size_t arrsize,
                               bool hasnan = false,
                               sort_order order = sort_order::sort_ascending);
    // key-value quicksort
    template <typename T1, typename T2>
    XSS_EXPORT_SYMBOL void
    keyvalue_qsort(T1 *key, T2 *val, size_t arrsize, bool hasnan = false);
    // quickselect
    template <typename T>
    XSS_HIDE_SYMBOL void qselect(T *arr,
                                 size_t k,
                                 size_t arrsize,
                                 bool hasnan = false,
                                 sort_order order = sort_order::sort_ascending);
    // partial sort
    template <typename T>
    XSS_HIDE_SYMBOL void partial_qsort(T *arr,
                                       size_t k,
                                       size_t arrsize,
                                       bool hasnan = false,
                                       sort_order order
                                       = sort_order::sort_ascending);
    // argsort
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argsort(T *arr, size_t arrsize, bool hasnan = false);
    // argselect
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
} // namespace avx2
namespace scalar {
    // quicksort
    template <typename T>
    XSS_HIDE_SYMBOL void
    qsort(T *arr, size_t arrsize, bool hasnan = false, bool descending = false);
    // key-value quicksort
    template <typename T1, typename T2>
    XSS_EXPORT_SYMBOL void
    keyvalue_qsort(T1 *key, T2 *val, size_t arrsize, bool hasnan = false);
    // quickselect
    template <typename T>
    XSS_HIDE_SYMBOL void qselect(T *arr,
                                 size_t k,
                                 size_t arrsize,
                                 bool hasnan = false,
                                 bool descending = false);
    // partial sort
    template <typename T>
    XSS_HIDE_SYMBOL void partial_qsort(T *arr,
                                       size_t k,
                                       size_t arrsize,
                                       bool hasnan = false,
                                       bool descending = false);
    // argsort
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argsort(T *arr, size_t arrsize, bool hasnan = false);
    // argselect
    template <typename T>
    XSS_HIDE_SYMBOL std::vector<size_t>
    argselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
} // namespace scalar
} // namespace xss
#endif
