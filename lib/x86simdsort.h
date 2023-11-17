#ifndef X86_SIMD_SORT
#define X86_SIMD_SORT
#include <stdint.h>
#include <vector>
#include <cstddef>

#define XSS_EXPORT_SYMBOL __attribute__((visibility("default")))
#define XSS_HIDE_SYMBOL __attribute__((visibility("hidden")))
#define UNUSED(x) (void)(x)

namespace x86simdsort {

// quicksort
template <typename T>
XSS_EXPORT_SYMBOL void qsort(T *arr, size_t arrsize, bool hasnan = false);

// quickselect
template <typename T>
XSS_EXPORT_SYMBOL void
qselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);

// partial sort
template <typename T>
XSS_EXPORT_SYMBOL void
partial_qsort(T *arr, size_t k, size_t arrsize, bool hasnan = false);

// argsort
template <typename T>
XSS_EXPORT_SYMBOL std::vector<size_t>
argsort(T *arr, size_t arrsize, bool hasnan = false);

// argselect
template <typename T>
XSS_EXPORT_SYMBOL std::vector<size_t>
argselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);

// argselect
template <typename T1, typename T2>
XSS_EXPORT_SYMBOL void
keyvalue_qsort(T1 *key, T2* val, size_t arrsize, bool hasnan = false);

} // namespace x86simdsort
#endif
