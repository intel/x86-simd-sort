#include "avx512-32bit-qsort.hpp"

int main() {
    const int size = 1000;
    float arr[size];
    avx512_qsort(arr, size);
    avx512_qselect(arr, 10, size);
    avx512_partial_qsort(arr, 10, size);
    return 0;
}
