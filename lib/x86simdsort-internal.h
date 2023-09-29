#ifndef XSS_ALL_METHODS
#define XSS_ALL_METHODS
#include <stdint.h>
#include <vector>

namespace xss {
namespace avx512 {
    // quicksort
    template <typename T>
    void qsort(T *arr, size_t arrsize);
    // quickselect
    template <typename T>
    void qselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // partial sort
    template <typename T>
    void partial_qsort(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // argsort
    template <typename T>
    std::vector<size_t> argsort(T *arr, size_t arrsize);
    // argselect
    template <typename T>
    std::vector<size_t> argselect(T *arr, size_t k, size_t arrsize);
} // namespace avx512
namespace avx2 {
    // quicksort
    template <typename T>
    void qsort(T *arr, size_t arrsize);
    // quickselect
    template <typename T>
    void qselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // partial sort
    template <typename T>
    void partial_qsort(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // argsort
    template <typename T>
    std::vector<size_t> argsort(T *arr, size_t arrsize);
    // argselect
    template <typename T>
    std::vector<size_t> argselect(T *arr, size_t k, size_t arrsize);
} // namespace avx2
namespace scalar {
    // quicksort
    template <typename T>
    void qsort(T *arr, size_t arrsize);
    // quickselect
    template <typename T>
    void qselect(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // partial sort
    template <typename T>
    void partial_qsort(T *arr, size_t k, size_t arrsize, bool hasnan = false);
    // argsort
    template <typename T>
    std::vector<size_t> argsort(T *arr, size_t arrsize);
    // argselect
    template <typename T>
    std::vector<size_t> argselect(T *arr, size_t k, size_t arrsize);
} // namespace scalar
} // namespace xss
#endif
