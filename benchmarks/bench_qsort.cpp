#include <benchmark/benchmark.h>
#include "rand_array.h"
#include "cpuinfo.h"
#include "avx512-16bit-qsort.hpp"
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"

template <typename T>
static void avx512_qsort(benchmark::State& state) {
    if (!cpu_has_avx512bw()) {
        state.SkipWithMessage("Requires AVX512 BW ISA");
    }
    if ((sizeof(T) == 2) && (!cpu_has_avx512_vbmi2())) {
        state.SkipWithMessage("Requires AVX512 VBMI2 ISA");
    }
    // Perform setup here
    size_t ARRSIZE = state.range(0);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    /* Initialize elements is reverse order */
    arr = get_uniform_rand_array<T>(ARRSIZE);
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        avx512_qsort<T>(arr.data(), ARRSIZE);
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

template <typename T>
static void stdsort(benchmark::State& state) {
    // Perform setup here
    size_t ARRSIZE = state.range(0);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    /* Initialize elements is reverse order */
    arr = get_uniform_rand_array<T>(ARRSIZE);
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        std::sort(arr.begin(), arr.end());
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

// Register the function as a benchmark
BENCHMARK(avx512_qsort<float>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<float>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<uint32_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<uint32_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<int32_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<int32_t>)->Arg(10000)->Arg(1000000);

BENCHMARK(avx512_qsort<double>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<double>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<uint64_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<uint64_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<int64_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<int64_t>)->Arg(10000)->Arg(10000000);

//BENCHMARK(avx512_qsort<float16>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<uint16_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<uint16_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(avx512_qsort<int16_t>)->Arg(10000)->Arg(1000000);
BENCHMARK(stdsort<int16_t>)->Arg(10000)->Arg(10000000);
// Run the benchmark
BENCHMARK_MAIN();
