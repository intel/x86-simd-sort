#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <span>
#include <chrono>
#include <map>
#include <set>
#include <ranges>
#include <utility>

#include "x86simdsort-c-api.h"

// if you want to test boost pdq_sort, you need to have boost installed and provide the include folder path to the compiler
#if __has_include(<boost/sort/sort.hpp>)
#define TEST_BOOST
#include <boost/sort/pdqsort/pdqsort.hpp>
constexpr bool hasBoost = true;
#else
constexpr bool hasBoost = false;
#endif

template <typename T>
void printVec(const std::vector<T>& v)
{
    for (const auto& elem : v)
        std::cout << elem << " ";
    std::cout << std::endl;
}

// name traits
template <typename T> struct Name;
template <> struct Name<int32_t>  { static constexpr const char* value = "int32_t"; };
template <> struct Name<uint32_t> { static constexpr const char* value = "uint32_t"; };
template <> struct Name<int64_t>  { static constexpr const char* value = "int64_t"; };
template <> struct Name<uint64_t> { static constexpr const char* value = "uint64_t"; };
template <> struct Name<float>    { static constexpr const char* value = "float"; };
template <> struct Name<double>   { static constexpr const char* value = "double"; };

enum SortAlgo { STL, PDQ, SIMD, ALGO_COUNT };
const char *algoNames[ALGO_COUNT] = {"stl", "pdq", "simd"};

enum SortFun { SORT, NTH, PARTIAL, KVSORT, KVNTH, KVPARTIAL, FUN_COUNT };
const char *funNames[FUN_COUNT] = {"sort", "nth", "partial",  "kvsort", "kvnth", "kvpartial"};

template <SortAlgo A>
using Algo = std::integral_constant<SortAlgo, A>;

template <SortFun F, SortAlgo A, bool WS = false>
struct AlgoFun {};

template <typename T>
std::string sortMsg(SortFun f, SortAlgo a, bool desc)
{
    std::ostringstream os;
    os << funNames[f] << '-' << algoNames[a] << ", Descend: " << desc
       << ", type: " << Name<T>::value;
    return os.str();
}

template <typename T>
std::string violation(const std::vector<T>& X, size_t i, size_t j, bool desc)
{
    std::ostringstream os;
    auto show = [&](auto k) {
        return (std::ostringstream{} << "X" << "[" << k << "](=" << X[k] << ")").str();
    };
    os << "\nordering violation: " << show(i) << (desc ? " >= " : " <= ") << show(j);
    return os.str();
}

template <typename T>
std::string violation(const std::vector<T> &X, const std::vector<uint32_t> &J, size_t i, size_t j, bool desc)
{
    std::ostringstream os;
    auto show = [&](auto k) {
        return (std::ostringstream{} << "X" << "[J[" << k << "](= " << J[k] << ")]=(" << X[J[k]] << ")").str();
    };
    os << "\nordering violation: " << show(i) << (desc ? " >= " : " <= ")
       << show(j);
    return os.str();
}

template <typename T>
class LocalCopy
{
    std::vector<T> m_ws;
    T *m_data;

public:
    LocalCopy(const std::vector<T> &src, T *ws)
        : m_data(nullptr)
    {
        if (!ws) {
            m_ws.assign(src.begin(), src.end());
            m_data = m_ws.data();
        }
        else {
            m_data = ws;
            std::copy(src.begin(), src.end(), m_data);
        }
    }

    T *data() { return m_data; }
};


template <SortFun F, SortAlgo A, typename T, bool Descend, typename = void>
struct Sorter;

template <typename, typename = void>
struct is_sorter_defined : std::false_type {};

// SFINAE-friendly detection if a specialization of Sorter is defined
template <SortFun F, SortAlgo A, typename T, bool Descend>
struct is_sorter_defined<Sorter<F, A, T, Descend>, std::void_t<decltype(&Sorter<F, A, T, Descend>::sort)>> : std::true_type {};

template <SortFun F, SortAlgo A, typename T, bool Descend>
inline constexpr bool is_sorter_defined_v = is_sorter_defined<Sorter<F, A, T, Descend>>::value;

template <SortAlgo A, typename T, bool Descend>
struct Sorter<SORT, A, T, Descend, std::enable_if_t<A == STL || A == SIMD || (hasBoost && A == PDQ)>>
{
    static void check(const std::vector<T>& X)
    {
        auto it = std::adjacent_find(X.begin(), X.end(), [](auto a, auto b) {
            return Descend ? a < b : a > b;
        });
        if (it != X.end()) {
            auto i = std::distance(X.begin(), it);
            std::cout << sortMsg<T>(SORT, A, Descend)
                      << violation(X, i, i + 1, Descend)
                      << std::endl;
            //printVec(X);
            std::exit(-1);
        }
    }

    static void sort(std::vector<T>& X)
    {
        const size_t n = X.size();

        if constexpr (A == STL) {
            std::sort(X.begin(), X.end(), [](auto a, auto b) {
                return Descend ? a > b : a < b;
            });
        }
#ifdef TEST_BOOST
        else if constexpr (hasBoost && A == PDQ) {
            boost::sort::pdqsort_branchless(X.begin(), X.end(), [](auto a, auto b) {
                return Descend ? a > b : a < b;
            });
        }
#endif
        else if constexpr (A == SIMD) {
            if constexpr (std::is_floating_point_v<T>)
                xss::qsort(X.data(), n, false, Descend);
            else
                xss::qsort(X.data(), n, Descend);
        }
        else {
            static_assert(false);
        }
    }
};


template <SortAlgo A, typename T, bool Descend>
struct Sorter<NTH, A, T, Descend, std::enable_if_t<A == SIMD || A == STL>>
{
    static void check(const std::vector<T> &X, size_t k)
    {
        auto pivot = X[k];
        size_t i = 0, j = 0;
        bool error = false;

        auto itLeft = std::find_if(X.begin(), X.begin() + k, [pivot](auto a) {
            return Descend ? a < pivot : a > pivot;
        });
        if (itLeft != X.begin() + k) {
            error = true;
            i = std::distance(X.begin(), itLeft);
            j = k;
        }
        else {
            auto itRight
                    = std::find_if(X.begin() + k + 1, X.end(), [pivot](auto a) {
                          return Descend ? a > pivot : a < pivot;
                      });
            if (itRight != X.end()) {
                error = true;
                i = k;
                j = std::distance(X.begin(), itRight);
            }
        }

        if (error) {
            std::cout << sortMsg<T>(NTH, A, Descend)
                      << violation(X, i, j, Descend)
                      << std::endl;
            //printVec(X);
            std::exit(-1);
        }
    }

    static void sort(std::vector<T> &X, size_t k)
    {
        const size_t n = X.size();

        if constexpr (A == STL) {
            std::nth_element(X.begin(), X.begin() + k, X.end(), [](auto a, auto b) {
                return Descend ? a > b : a < b;
            });
        }
        else if constexpr (A == SIMD) {
            if constexpr (std::is_floating_point_v<T>)
                xss::qselect(X.data(), k, n, false, Descend);
            else
                xss::qselect(X.data(), k, n, Descend);
        }
        else {
            static_assert(false);
        }
    }
};

template <SortAlgo A, typename T, bool Descend>
struct Sorter<PARTIAL, A, T, Descend, std::enable_if_t<A == SIMD || A == STL>> {
    static void check(const std::vector<T> &X, size_t k)
    {
        size_t i = 0, j = 0;
        bool error = false;

        auto itLeft = std::adjacent_find(X.begin(), X.begin() + k, [](auto a, auto b) {
            return Descend ? a < b : a > b;
        });
        if (itLeft != X.begin() + k) {
            error = true;
            i = std::distance(X.begin(), itLeft);
            j = i + 1;
        }
        else {
            auto pivot = X[k - 1];
            auto itRight = std::find_if(X.begin() + k, X.end(), [pivot](auto a) {
                return Descend ? a > pivot : a < pivot;
            });
            if (itRight != X.end()) {
                error = true;
                i = k - 1;
                j = std::distance(X.begin(), itRight);
            }
        }

        if (error) {
            std::cout << sortMsg<T>(PARTIAL, A, Descend)
                      << violation(X, i, j, Descend) << std::endl;
            //printVec(X);
            std::exit(-1);
        }
    }

    static void sort(std::vector<T> &X, size_t k)
    {
        const size_t n = X.size();

        if constexpr (A == STL) {
            std::partial_sort(
                    X.begin(), X.begin() + k, X.end(), [](auto a, auto b) {
                        return Descend ? a > b : a < b;
                    });
        }
        else if constexpr (A == SIMD) {
            if constexpr (std::is_floating_point_v<T>)
                xss::partial_qsort(X.data(), k, n, false, Descend);
            else
                xss::partial_qsort(X.data(), k, n, Descend);
        }
        else {
            static_assert(false);
        }
    }
};

template <SortAlgo A, typename T, bool Descend>
struct Sorter<KVSORT, A, T, Descend, std::enable_if_t<A == STL || A == SIMD || (hasBoost && A == PDQ)>>
{
    static void check(const std::vector<T>& X, const std::vector<uint32_t>& J, T *)
    {
        auto it = std::adjacent_find(J.begin(), J.end(), [&X](auto i, auto j) {
            return Descend ? X[i] < X[j] : X[i] > X[j];
        });
        if (it != J.end()) {
            auto i = std::distance(J.begin(), it);
            std::cout << sortMsg<T>(KVSORT, A, Descend)
                      << violation(X, J, i, i + 1, Descend)
                      << std::endl;
            //printVec(X);
            //printVec(J);
            std::exit(-1);
        }
    }

    static void sort(const std::vector<T> &X, std::vector<uint32_t>& J, T *ws)
    {
        std::iota(J.begin(), J.end(), 0);

        if constexpr (A == STL) {
            std::sort(J.begin(), J.end(), [&X](auto i, auto j) {
                return Descend ? X[i] > X[j] : X[i] < X[j];
            });
        }
#ifdef TEST_BOOST
        else if constexpr (hasBoost && A == PDQ) {
            boost::sort::pdqsort(J.begin(), J.end(), [&X](auto i, auto j) {
                return Descend ? X[i] > X[j] : X[i] < X[j];
            });
        }
#endif
        else if constexpr (A == SIMD) {
            const size_t n = X.size();

            LocalCopy keys(X, ws);

            if constexpr (std::is_floating_point_v<T>)
                xss::keyvalue_qsort(keys.data(), J.data(), n, false, false);
            else
                xss::keyvalue_qsort(keys.data(), J.data(), n, false);

            if (Descend)
                std::ranges::reverse(J);
        }
        else {
            static_assert(false);
        }
    }
};

template <SortAlgo A, typename T, bool Descend>
struct Sorter<KVNTH, A, T, Descend, std::enable_if_t<A == SIMD || A == STL>> {
    static void check(const std::vector<T> &X,
                      const std::vector<uint32_t> &J,
                      size_t k,
                      T *)
    {
        bool error = false;
        size_t i = 0, j = 0;
        auto pivot = X[J[k]];

        auto itLeft = std::find_if(J.begin(), J.begin() + k, [&](auto j) {
            return Descend ? X[j] < pivot : X[j] > pivot;
        });
        if (itLeft != J.begin() + k) {
            error = true;
            i = std::distance(J.begin(), itLeft);
            j = k;
        }
        else {
            auto itRight
                    = std::find_if(J.begin() + k + 1, J.end(), [&](auto j) {
                          return Descend ? X[j] > pivot : X[j] < pivot;
                      });
            if (itRight != J.end()) {
                error = true;
                i = k;
                j = std::distance(J.begin(), itRight);
            }
        }

        if (error) {
            std::cout << sortMsg<T>(KVNTH, A, Descend)
                      << violation(X, J, i, j, Descend)
                      << std::endl;
            //printVec(X);
            std::exit(-1);
        }
    }

    static void sort(std::vector<T> &X, std::vector<uint32_t> &J, size_t k, T* ws)
    {
        std::iota(J.begin(), J.end(), 0);

        if constexpr (A == STL) {
            std::nth_element(
                    X.begin(), X.begin() + k, X.end(), [](auto a, auto b) {
                        return Descend ? a > b : a < b;
                    });
        }
        else if constexpr (A == SIMD) {
            const size_t n = X.size();

            LocalCopy keys(X, ws);

            if constexpr (std::is_floating_point_v<T>)
                xss::keyvalue_qselect(keys.data(), J.data(), k, n, false, Descend);
            else
                xss::keyvalue_qselect(keys.data(), J.data(), k, n, Descend);
        }
        else {
            static_assert(false);
        }
    }
};

template <SortAlgo A, typename T, bool Descend>
struct Sorter<KVPARTIAL, A, T, Descend, std::enable_if_t<A == SIMD || A == STL>> {
    static void check(const std::vector<T> &X,
                      const std::vector<uint32_t> &J,
                      size_t k,
                      T *)
    {
        bool error = false;
        size_t i = 0, j = 0;

        auto itLeft = std::adjacent_find(J.begin(), J.begin() + k, [&X](auto j, auto p) {
            return Descend ? X[j] < X[p] : X[j] > X[p];
        });
        if (itLeft != J.begin() + k) {
            error = true;
            i = std::distance(J.begin(), itLeft);
            j = i + 1;
        }
        else {
            auto pivot = X[J[k - 1]];
            auto itRight
                    = std::find_if(J.begin() + k, J.end(), [&](auto j) {
                          return Descend ? X[j] > pivot : X[j] < pivot;
                      });
            if (itRight != J.end()) {
                error = true;
                i = k - 1;
                j = std::distance(J.begin(), itRight);
            }
        }

        if (error) {
            std::cout << sortMsg<T>(KVPARTIAL, A, Descend)
                      << violation(X, J, i, j, Descend) << std::endl;
            //printVec(X);
            std::exit(-1);
        }
    }

    static void sort(std::vector<T> &X, std::vector<uint32_t> &J, size_t k, T *ws)
    {
        std::iota(J.begin(), J.end(), 0);

        if constexpr (A == STL) {
            std::partial_sort(X.begin(), X.begin() + k, X.end(), [](auto a, auto b) {
                return Descend ? a > b : a < b;
            });
        }
        else if constexpr (A == SIMD) {
            const size_t n = X.size();

            LocalCopy keys(X, ws);

            if constexpr (std::is_floating_point_v<T>)
                xss::keyvalue_partial_qsort(keys.data(), J.data(), k, n, false, Descend);
            else
                xss::keyvalue_partial_qsort(keys.data(), J.data(), k, n, Descend);
        }
        else {
            static_assert(false);
        }
    }
};

// compare performance of sort_simd vs sort_classic for keys of type T
// use vectors of size 4, 8, 16, 32, ...
// report results in tabular format
template <bool Descend, typename T>
void benchmark()
{
    using namespace std::chrono;

    struct Key {
        SortFun f;
        SortAlgo a;
        bool ws;
        size_t n;
        bool operator<(const Key& o) const {
            return std::tie(f,a,ws,n) < std::tie(o.f,o.a,o.ws,o.n);
        }
    };

    std::map<Key, double> times;

    auto runTest = [&]<SortFun F, SortAlgo A, bool UseWS, typename...Ts>(AlgoFun<F,A,UseWS>&&, std::vector<T>& x, Ts&&...args) {
        if constexpr (is_sorter_defined_v<F,A,T,Descend>) {
            auto tStart = high_resolution_clock::now();
            Sorter<F, A, T, Descend>::sort(x, std::forward<Ts>(args)...);
            auto tEnd = high_resolution_clock::now();
            auto [it, _] = times.insert({{F, A, UseWS, x.size()}, 0.0});
            it->second += duration_cast<microseconds>(tEnd - tStart).count();
            Sorter<F, A, T, Descend>::check(x, std::forward<Ts>(args)...);
        }
    };

    std::cout << "TESTING: type: " << Name<T>::value
              << ", Order: " << (Descend ? "descending" : "ascending") << "\n"; \

    constexpr size_t max_exp = 20;
    for (size_t _n = 4; _n <= 1<<20; _n *= 2) {
        size_t m = std::min<size_t>(std::max<size_t>((1 << max_exp) / _n, 2), 200);
        for (auto n : {_n, 3*_n/2}) {

            std::vector<T> X(n), Xcopy(n);
            std::vector<uint32_t> J(n);
            std::vector<T> ws(n);

            auto sortTest = [&]<SortAlgo A>(Algo<A>&&) {
                std::ranges::copy(X, Xcopy.begin());
                runTest(AlgoFun<SORT, A> {}, Xcopy);
            };

            auto nthTest = [&]<SortAlgo A>(Algo<A> &&) {
                std::ranges::copy(X, Xcopy.begin());
                runTest(AlgoFun<NTH, A> {}, Xcopy, n/2);
            };

            auto partialTest = [&]<SortAlgo A>(Algo<A> &&) {
                std::ranges::copy(X, Xcopy.begin());
                runTest(AlgoFun<PARTIAL, A> {}, Xcopy, n / 2);
            };

            auto kvTest = [&]<SortAlgo A>(Algo<A>&&) {
                runTest(AlgoFun<KVSORT, A, false> {}, X, J, nullptr);
                if constexpr (A == SIMD)
                    runTest(AlgoFun<KVSORT, A, true> {}, X, J, ws.data());
            };

            auto kvnthTest = [&]<SortAlgo A>(Algo<A> &&) {
                runTest(AlgoFun<KVNTH, A, false> {}, X, J, n/2, nullptr);
                if constexpr (A == SIMD)
                    runTest(AlgoFun<KVNTH, A, true> {}, X, J, n/2, ws.data());
            };

            auto kvpartialTest = [&]<SortAlgo A>(Algo<A> &&) {
                runTest(AlgoFun<KVPARTIAL, A, false> {}, X, J, n / 2, nullptr);
                if constexpr (A == SIMD)
                    runTest(AlgoFun<KVPARTIAL, A, true> {}, X, J, n / 2, ws.data());
            };

            auto allAlgoTests = [&]<size_t...As>(std::index_sequence<As...>&&) {
                (sortTest(Algo<(SortAlgo)As>{}), ...);
                (nthTest(Algo<(SortAlgo)As> {}), ...);
                (partialTest(Algo<(SortAlgo)As> {}), ...);
                (kvTest(Algo<(SortAlgo)As>{}), ...);
                (kvnthTest(Algo<(SortAlgo)As> {}), ...);
                (kvpartialTest(Algo<(SortAlgo)As> {}), ...);
            };

            for (size_t rep = 0; rep < m; ++rep) {
                // Fill with random data for each repetition
                for (auto &v : X)
                    v = static_cast<T>(std::rand());

                allAlgoTests(std::make_index_sequence<ALGO_COUNT>{});
            }

            // take the average time over m repetitions
            for (auto &[k, t] : times)
                if (k.n == n)
                    t /= m;
        }
    }

    // print timing results in tabular format

    // extracts unique labels and sizes from timing results
    std::set<std::tuple<SortFun, SortAlgo, bool>> labels;
    std::set<size_t> sizes;
    for (auto [f, a, ws, n] : times | std::views::keys) {
        labels.insert({f, a, ws});
        sizes.insert(n);
    }

    // print labels
    const int colSpacing = 18;
    std::cout << std::setw(8) << "Size";
    for (auto [f, a, ws] : labels) {
        std::ostringstream label;
        label << funNames[f] << '-' << algoNames[a] << (ws ? "-WS" : "");
        std::cout << std::setw(colSpacing) << label.str();
    }
    std::cout << std::endl;

    // print times
    for (auto n : sizes) {
        std::cout << std::setw(8) << n;
        for (auto [f, a, ws] : labels) {
            auto it = times.find({f, a, ws, n});
            if (it != times.end())
                std::cout << std::setw(colSpacing) << std::fixed << std::setprecision(2) << it->second;
            else
                std::cout << std::setw(colSpacing) << "N/A";
        }
        std::cout << std::endl;
    }
}

template <bool Descend>
void benchmarks()
{
    benchmark<Descend,uint32_t>();
    benchmark<Descend,uint64_t>();
    benchmark<Descend,int32_t>();
    benchmark<Descend,int64_t>();
    benchmark<Descend,float>();
    benchmark<Descend,double>();
}


int main()
{
    benchmarks<true>();
    benchmarks<false>();

    return 0;
}

