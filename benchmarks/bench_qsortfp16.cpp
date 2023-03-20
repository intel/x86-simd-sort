#include <benchmark/benchmark.h>
#include "rand_array.h"
#include "cpuinfo.h"
#include "avx512fp16-16bit-qsort.hpp"

template <typename T>
static void avx512_qsort(benchmark::State& state) {
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        size_t ARRSIZE = state.range(0);
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        /* Initialize elements */
        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float) rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call avx512 quicksort */
        for (auto _ : state) {
            avx512_qsort<T>(arr.data(), ARRSIZE);
            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

template <typename T>
static void stdsort(benchmark::State& state) {
    if (cpu_has_avx512fp16()) {
        // Perform setup here
        size_t ARRSIZE = state.range(0);
        std::vector<T> arr;
        std::vector<T> arr_bkp;

        for (size_t jj = 0; jj < ARRSIZE; ++jj) {
            _Float16 temp = (float) rand() / (float)(RAND_MAX);
            arr.push_back(temp);
        }
        arr_bkp = arr;

        /* call avx512 quicksort */
        for (auto _ : state) {
            std::sort(arr.begin(), arr.end());
            state.PauseTiming();
            arr = arr_bkp;
            state.ResumeTiming();
        }
    }
    else {
        state.SkipWithMessage("Requires AVX512-FP16 ISA");
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_qsort<_Float16>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<_Float16>)->Arg(10000)->Arg(1000000);
