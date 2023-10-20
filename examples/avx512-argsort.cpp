#include "avx512-64bit-argsort.hpp"

int main() {
    const int size = 1000;
    float arr[size];
    std::vector<size_t> arg1 = avx512_argsort(arr, size);
    std::vector<size_t> arg2 = avx512_argselect(arr, 10, size);
    return 0;
}
