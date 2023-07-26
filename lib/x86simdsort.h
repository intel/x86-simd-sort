#ifndef X86_SIMD_SORT
#define X86_SIMD_SORT
#include <stdint.h>
#include <vector>

namespace x86simdsort {
// quicksort
template <typename T>
void qsort(T *arr, int64_t arrsize);
// quickselect
template <typename T>
void qselect(T *arr, int64_t k, int64_t arrsize, bool hasnan = false);
// partial sort
template <typename T>
void partial_qsort(T *arr, int64_t k, int64_t arrsize, bool hasnan = false);
// argsort
template <typename T>
std::vector<int64_t> argsort(T *arr, int64_t arrsize);
// argselect
template <typename T>
std::vector<int64_t> argselect(T *arr, int64_t k, int64_t arrsize);
} // namespace x86simdsort
#endif
