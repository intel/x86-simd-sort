#include <limits>
#include <cmath>

typedef union {
    _Float16 f_;
    uint16_t i_;
} Fp16Bits;

template <typename T>
bool isnan(T elem)
{
    return std::isnan(elem);
}

#ifdef __FLT16_MAX__
template <>
bool isnan<_Float16>(_Float16 elem)
{
    Fp16Bits temp;
    temp.f_ = elem;
    return (temp.i_ & 0x7c00) == 0x7c00;
}
#endif

template <typename T>
bool isunordered(T a, T b)
{
    return !isnan(a + b);
}

/*
 * Custom comparator class to handle NAN's: treats NAN  > INF
 */
template <typename T, typename Comparator>
struct compare {
    static constexpr auto op = Comparator {};
    bool operator()(const T a, const T b)
    {
#ifdef __FLT16_MAX__
        if constexpr ((std::is_floating_point_v<T>) || (std::is_same_v<T, _Float16>)) {
#else
        if constexpr (std::is_floating_point_v<T>) {
#endif
            T inf = std::numeric_limits<T>::infinity();
            T one = (T) 1.0;
            if (!isunordered(a, b)) { return op(a, b); }
            else if ((isnan(a)) && (!isnan(b))) {
                return b == inf ? op(inf, one) : op(inf, b);
            }
            else if ((!isnan(a)) && (isnan(b))) {
                return a == inf ? op(one, inf) : op(a, inf);
            }
            else {
                return op(one, one);
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
