#ifndef X86_SIMD_SORT
#define X86_SIMD_SORT
#include <stdint.h>
#include <vector>
#include <cstddef>

#define XSS_EXPORT_SYMBOL __attribute__((visibility("default")))
#define XSS_HIDE_SYMBOL __attribute__((visibility("hidden")))

namespace x86simdsort {
// quicksort
template <typename T>
XSS_EXPORT_SYMBOL void qsort(T *arr, size_t arrsize);
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
XSS_EXPORT_SYMBOL std::vector<size_t> argsort(T *arr, size_t arrsize);
// argselect
template <typename T>
XSS_EXPORT_SYMBOL std::vector<size_t>
argselect(T *arr, size_t k, size_t arrsize);
} // namespace x86simdsort
#endif
