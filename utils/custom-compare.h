#include <limits>
#include <cmath>
/*
 * Custom comparator class to handle NAN's: treats NAN  > INF
 */
template <typename T, typename Comparator>
struct compare {
    static constexpr auto op = Comparator {};
    bool operator()(const T a, const T b)
    {
        if constexpr (std::is_floating_point_v<T>) {
            T inf = std::numeric_limits<T>::infinity();
            if (!std::isunordered(a, b)) { return op(a, b); }
            else if ((std::isnan(a)) && (!std::isnan(b))) {
                return b == inf ? op(inf, 1.) : op(inf, b);
            }
            else if ((!std::isnan(a)) && (std::isnan(b))) {
                return a == inf ? op(1., inf) : op(a, inf);
            }
            else {
                return op(1., 1.);
            }
        }
        else {
            return op(a, b);
        }
    }
};

template <typename T, typename Comparator>
struct compare_arg {
    compare_arg(const T* arr)
    {
        this->arr = arr;
    }
    bool operator()(const int64_t a, const int64_t b)
    {
        return compare<T, Comparator>()(arr[a], arr[b]);
    }
    const T* arr;
};
