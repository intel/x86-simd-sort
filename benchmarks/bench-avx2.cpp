#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
#include "cpuinfo.h"
#include "rand_array.h"
#include <benchmark/benchmark.h>

#define MY_BENCHMARK_CAPTURE(func, T, test_case_name, ...) \
    BENCHMARK_PRIVATE_DECLARE(func) \
            = (::benchmark::internal::RegisterBenchmarkInternal( \
                    new ::benchmark::internal::FunctionBenchmark( \
                            #func "/" #test_case_name "/" #T, \
                            [](::benchmark::State &st) { \
                                func<T>(st, __VA_ARGS__); \
                            })))

#define BENCH(func, type) \
    MY_BENCHMARK_CAPTURE( \
            func, type, random_5k, 5000, std::string("random")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, random_100k, 100000, std::string("random")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, random_1m, 1000000, std::string("random")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, random_10m, 10000000, std::string("random")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, sorted_10k, 10000, std::string("sorted")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, constant_10k, 10000, std::string("constant")); \
    MY_BENCHMARK_CAPTURE( \
            func, type, reverse_10k, 10000, std::string("reverse"));

template <typename T, class... Args>
static void stdsort(benchmark::State &state, Args &&...args)
{
    auto args_tuple = std::make_tuple(std::move(args)...);
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(ARRSIZE); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1)[0];
        for (size_t ii = 0; ii < ARRSIZE; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
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

template <typename T, class... Args>
static void avx2qsort(benchmark::State &state, Args &&...args)
{
    auto args_tuple = std::make_tuple(std::move(args)...);
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<T> arr_bkp;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") { arr = get_uniform_rand_array<T>(ARRSIZE); }
    else if (arrtype == "sorted") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
    }
    else if (arrtype == "constant") {
        T temp = get_uniform_rand_array<T>(1)[0];
        for (size_t ii = 0; ii < ARRSIZE; ++ii) {
            arr.push_back(temp);
        }
    }
    else if (arrtype == "reverse") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
        std::sort(arr.begin(), arr.end());
        std::reverse(arr.begin(), arr.end());
    }
    arr_bkp = arr;

    /* call avx512 quicksort */
    for (auto _ : state) {
        avx2_qsort<T>(arr.data(), ARRSIZE);
        state.PauseTiming();
        arr = arr_bkp;
        state.ResumeTiming();
    }
}

#define BENCH_BOTH_QSORT(type)\
    BENCH(avx2qsort, type)\
    BENCH(stdsort, type)

//BENCH_BOTH_QSORT(uint16_t)
//BENCH_BOTH_QSORT(int16_t)
BENCH_BOTH_QSORT(uint32_t)
BENCH_BOTH_QSORT(int32_t)
BENCH_BOTH_QSORT(uint64_t)
BENCH_BOTH_QSORT(int64_t)
BENCH_BOTH_QSORT(float)
BENCH_BOTH_QSORT(double)


