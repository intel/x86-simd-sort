#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <span>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <limits>
#include <bit>

#include "x86simdsort-c-api.h"

template <typename T, typename U>
struct Workspace {
    std::vector<T> m_data;
    std::vector<U> m_index;

    // default constructor
    Workspace() = default;

    // constructor with size n
    Workspace(size_t n)
    {
        resize(n);
    }
    void resize(size_t n)
    {
        const size_t np = std::bit_ceil(n); // next power of 2
        m_data.resize(np);
        if (n < np)
            m_index.resize(np);
    }
};

template <typename T>
void printVec(const std::vector<T>& v)
{
    for (const auto& elem : v)
        std::cout << elem << " ";
    std::cout << std::endl;
}

template <bool Descend, typename T, typename U>
void checkSort(T &X, U &J)
{
    auto it = std::adjacent_find(J.begin(), J.end(), [&X](auto i, auto j) {
        return Descend ? X[i] < X[j] : X[i] > X[j];
    });
    if (it != J.end()) {
        std::cout << "Found adjacent elements where x[i] > x[j]: " << *it
                  << " and " << *(it + 1) << std::endl;
        printVec(X);
        printVec(J);
        std::exit(-1);
    }
}

template <bool Descend, typename T>
void sort_classic(const std::vector<T> &X, std::vector<uint32_t> J)
{
    std::iota(J.begin(), J.end(), uint32_t(0));
    std::sort(J.begin(), J.end(), [&X](auto i, auto j) {
        return Descend ? X[i] > X[j] : X[i] < X[j];
    });
}

template <bool Descend, typename T>
void sort_simd(const std::vector<T> &X,
               std::vector<uint32_t> J,
               Workspace<T, uint32_t> *ws)
{
    const size_t n = X.size();
    const size_t np = std::bit_ceil(n);

    Workspace<T, uint32_t> wslocal;
    if (!ws) {
        ws = &wslocal;
        wslocal.resize(n);
    }

    auto *keys = ws->m_data.data();
    auto *index = J.data();
    std::copy_n(X.data(), n, keys);

    if (n < np) {
        index = ws->m_index.data();
        std::fill_n(keys + n,
                    np - n,
                    Descend ? std::numeric_limits<T>::min()
                            : std::numeric_limits<T>::max());
    }

    std::iota(index, index + np, uint32_t(0));
    if constexpr (std::is_floating_point_v<T>)
        xss::keyvalue_qsort(keys, index, np, false, Descend);
    else
        xss::keyvalue_qsort(keys, index, np, Descend);

    if (n < np)
        std::copy_n(index, n, J.data());
}

// compare performance of sort_simd vs sort_classic for keys of type T
// use vectors of size 4, 8, 16, ..., 1024, 2048, 4096
// report results in tabular format
template <bool Descend, typename T>
void benchmark(bool useWorkspace)
{
    using namespace std::chrono;
    constexpr size_t max_exp = 18;
    std::cout << std::setw(8) << "Size" << std::setw(16) << "Classic (us)"
              << std::setw(16) << "SIMD (us)" << std::endl;

    for (size_t _n = 4; _n <= 4096; _n *= 2) {
        size_t m = (1 << max_exp) / _n;
        for (auto n : {_n, _n + 1}) {

            double classic_total = 0.0;
            double simd_total = 0.0;

            std::vector<T> X(n);
            std::vector<uint32_t> J(n);
            Workspace<T, uint32_t> ws(n);
            Workspace<T, uint32_t> *pws = useWorkspace ? &ws : nullptr;

            for (size_t rep = 0; rep < m; ++rep) {
                // Fill with random data for each repetition
                for (auto &v : X)
                    v = static_cast<T>(std::rand());

                // Classic sort timing
                auto start1 = high_resolution_clock::now();
                sort_classic<Descend>(X, J);
                auto end1 = high_resolution_clock::now();
                classic_total
                        += duration_cast<microseconds>(end1 - start1).count();
                checkSort<Descend>(X, J);

                // SIMD sort timing
                auto start2 = high_resolution_clock::now();
                sort_simd<Descend>(X, J, pws);
                auto end2 = high_resolution_clock::now();
                simd_total
                        += duration_cast<microseconds>(end2 - start2).count();
                checkSort<Descend>(X, J);
            }

            double classic_avg = classic_total / m;
            double simd_avg = simd_total / m;

            std::cout << std::setw(8) << n << std::setw(16) << std::fixed
                      << std::setprecision(2) << classic_avg << std::setw(16)
                      << std::fixed << std::setprecision(2) << simd_avg
                      << std::endl;
        }
    }
}

#define TEST(n, ty) \
    std::cout << "Use WS: " << useWS << ", type: " << #n << ", Descend: " << Descend << "\n"; \
    benchmark<Descend, ty>(useWS);

template <bool Descend>
void benchmarks(bool useWS)
{
    TEST(uint32, uint32_t)
    TEST(uint64, uint64_t)
    TEST(int32, int32_t)
    TEST(int64, int64_t)
    TEST(float, float)
    TEST(double, double)
}


int main()
{
    benchmarks<true>(true);
    benchmarks<false>(true);
    benchmarks<true>(false);
    benchmarks<false>(false);

    return 0;
}

