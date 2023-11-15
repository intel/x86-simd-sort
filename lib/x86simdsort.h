#ifndef X86_SIMD_SORT
#define X86_SIMD_SORT
#include <stdint.h>
#include <vector>
#include <cstddef>
#include <functional>

#define XSS_EXPORT_SYMBOL __attribute__((visibility("default")))
#define XSS_HIDE_SYMBOL __attribute__((visibility("hidden")))
#define UNUSED(x) (void)(x)

template <typename T>
XSS_HIDE_SYMBOL void permute_array_in_place(T *A, std::vector<size_t> P)
{
    for (size_t i = 0; i < P.size(); i++) {
        size_t curr = i;
        size_t next = P[curr];
        while (next != i) {
            std::swap(A[curr], A[next]);
            P[curr] = curr;
            curr = next;
            next = P[next];
        }
        P[curr] = curr;
    }
}

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

// keyvalue sort
template <typename T1, typename T2>
XSS_EXPORT_SYMBOL void
keyvalue_qsort(T1 *key, T2* val, size_t arrsize, bool hasnan = false);

// sort an object
template <typename T, typename F>
XSS_EXPORT_SYMBOL void object_qsort(T *arr, size_t arrsize, const F key_func)
{
    using return_type_of =
            typename decltype(std::function {key_func})::result_type;
    std::vector<return_type_of> keys(arrsize);
    for (size_t ii = 0; ii < arrsize; ++ii) {
        keys[ii] = key_func(arr[ii]);
    }
    std::vector<size_t> arg = x86simdsort::argsort(keys.data(), arrsize);
    permute_array_in_place(arr, arg);
}

} // namespace x86simdsort
#endif
