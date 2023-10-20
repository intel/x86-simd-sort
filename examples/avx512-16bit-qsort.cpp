#include "avx512-16bit-qsort.hpp"

int main() {
    const int size = 1000;
    short arr[size];
    avx512_qsort(arr, size);
    avx512_qselect(arr, 10, size);
    avx512_partial_qsort(arr, 10, size);
    return 0;
}
