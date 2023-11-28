#include "avx512-64bit-keyvaluesort.hpp"

int main() {
    const int size = 1000;
    int64_t arr1[size];
    uint64_t arr2[size];
    double arr3[size];
    float arr4[size];
    avx512_qsort_kv(arr1, arr1, size);
    avx512_qsort_kv(arr1, arr2, size);
    avx512_qsort_kv(arr1, arr3, size);
    avx512_qsort_kv(arr2, arr1, size);
    avx512_qsort_kv(arr2, arr2, size);
    avx512_qsort_kv(arr2, arr3, size);
    avx512_qsort_kv(arr3, arr1, size);
    avx512_qsort_kv(arr3, arr2, size);
    avx512_qsort_kv(arr1, arr4, size);
    avx512_qsort_kv(arr2, arr4, size);
    avx512_qsort_kv(arr3, arr4, size);
    return 0;
    return 0;
}
