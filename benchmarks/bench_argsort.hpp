#include "bench-qsort-common.h"

template<typename T>
std::vector<int64_t> stdargsort(const std::vector<T> &array) {
    std::vector<int64_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int64_t left, int64_t right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

#define MY_BENCHMARK_CAPTURE(func, T, test_case_name, ...)     \
  BENCHMARK_PRIVATE_DECLARE(func) =                      \
      (::benchmark::internal::RegisterBenchmarkInternal( \
          new ::benchmark::internal::FunctionBenchmark(  \
              #func "/" #test_case_name "/" #T,                 \
              [](::benchmark::State& st) { func<T>(st, __VA_ARGS__); })))

template <typename T, class ...Args>
static void stdargsort(benchmark::State& state, Args&&... args) {
    auto args_tuple = std::make_tuple(std::move(args)...);
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<int64_t> inx;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
    }
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

    /* call avx512 quicksort */
    for (auto _ : state) {
        inx = stdargsort(arr);
    }
}

template <typename T, class ...Args>
static void avx512argsort(benchmark::State& state, Args&&... args) {
    auto args_tuple = std::make_tuple(std::move(args)...);
    if (!cpu_has_avx512bw()) {
        state.SkipWithMessage("Requires AVX512 BW ISA");
    }
    // Perform setup here
    size_t ARRSIZE = std::get<0>(args_tuple);
    std::vector<T> arr;
    std::vector<int64_t> inx;

    std::string arrtype = std::get<1>(args_tuple);
    if (arrtype == "random") {
        arr = get_uniform_rand_array<T>(ARRSIZE);
    }
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

    /* call avx512 quicksort */
    for (auto _ : state) {
        inx = avx512_argsort<T>(arr.data(), ARRSIZE);
    }
}

#define BENCH(func, type)\
MY_BENCHMARK_CAPTURE(func, type, random_10000, 10000, std::string("random")); \
MY_BENCHMARK_CAPTURE(func, type, random_100000, 100000, std::string("random")); \
MY_BENCHMARK_CAPTURE(func, type, sorted_10000, 100000, std::string("sorted")); \
MY_BENCHMARK_CAPTURE(func, type, constant_10000, 100000, std::string("constant")); \
MY_BENCHMARK_CAPTURE(func, type, reverse_10000, 100000, std::string("reverse")); \

BENCH(avx512argsort, int64_t)
BENCH(stdargsort, int64_t)