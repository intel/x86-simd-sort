#include "x86simdsort.h"
#include "rand_array.h"
#include <benchmark/benchmark.h>

#ifdef __FLT16_MAX__
template <>
std::vector<_Float16> get_uniform_rand_array(
        int64_t arrsize,
        _Float16 max,
        _Float16 min)
{
    (void)(max); (void)(min);
    std::vector<_Float16> arr;
    for (auto jj = 0; jj < arrsize; ++jj) {
        _Float16 temp = (float)rand() / (float)(RAND_MAX);
        arr.push_back(temp);
    }
    return arr;
}
#endif

#define MY_BENCHMARK_CAPTURE(func, T, test_case_name, ...) \
    BENCHMARK_PRIVATE_DECLARE(func) \
            = (::benchmark::internal::RegisterBenchmarkInternal( \
                    new ::benchmark::internal::FunctionBenchmark( \
                            #func "/" #test_case_name "/" #T, \
                            [](::benchmark::State &st) { \
                                func<T>(st, __VA_ARGS__); \
                            })))

#define BENCH_SORT(func, type) \
    MY_BENCHMARK_CAPTURE(func, type, smallrandom_128, 128, std::string("random")); \
    MY_BENCHMARK_CAPTURE(func, type, smallrandom_256, 256, std::string("random")); \
    MY_BENCHMARK_CAPTURE(func, type, smallrandom_512, 512, std::string("random")); \
    MY_BENCHMARK_CAPTURE(func, type, smallrandom_1k, 1024, std::string("random")); \
    MY_BENCHMARK_CAPTURE(func, type, random_5k, 5000, std::string("random")); \
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

#define BENCH_PARTIAL(func, type) \
    MY_BENCHMARK_CAPTURE(func, type, k10, 10000, 10); \
    MY_BENCHMARK_CAPTURE(func, type, k100, 10000, 100); \
    MY_BENCHMARK_CAPTURE(func, type, k1000, 10000, 1000); \
    MY_BENCHMARK_CAPTURE(func, type, k5000, 10000, 5000); \

#include "bench-argsort.hpp"
#include "bench-partial-qsort.hpp"
#include "bench-qselect.hpp"
#include "bench-qsort.hpp"

