#include "avx2-32bit-qsort.hpp"

int main() {
    const int size = 1000;
    float arr[size];
    avx2_qsort(arr, size);
    avx2_qselect(arr, 10, size);
    avx2_partial_qsort(arr, 10, size);
    return 0;
}
