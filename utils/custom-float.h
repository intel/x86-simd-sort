#ifndef UTILS_FLOAT
#define UTILS_FLOAT
namespace xss {
namespace fp
{
    template <typename T>
    inline constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

    template <typename T>
    bool isnan(T elem)
    {
        return std::isnan(elem);
    }
    template <typename T>
    bool isunordered(T a, T b)
    {
        return std::isunordered(a, b);
    }
    template <typename T>
    T max()
    {
        return std::numeric_limits<T>::max();
    }
    template <typename T>
    T min()
    {
        return std::numeric_limits<T>::min();
    }
    template <typename T>
    T infinity()
    {
        return std::numeric_limits<T>::infinity();
    }
    template <typename T>
    T quiet_NaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

#ifdef __FLT16_MAX__
    typedef union {
        _Float16 f_;
        uint16_t i_;
    } Fp16Bits;

    _Float16 convert_bits(uint16_t val)
    {
        Fp16Bits temp;
        temp.i_ = val;
        return temp.f_;
    }

    template <>
    inline constexpr bool is_floating_point_v<_Float16> = true;

    template <>
    bool isnan<_Float16>(_Float16 elem)
    {
        return elem != elem;
    }
    template <>
    bool isunordered<_Float16>(_Float16 a, _Float16 b)
    {
        return isnan(a) || isnan(b);
    }
    template <>
    _Float16 max<_Float16>()
    {
        return convert_bits(0x7bff);
    }
    template <>
    _Float16 min<_Float16>()
    {
        return convert_bits(0x0400);
    }
    template <>
    _Float16 infinity<_Float16>()
    {
        return convert_bits(0x7c00);
    }
    template <>
    _Float16 quiet_NaN<_Float16>()
    {
        return convert_bits(0x7c01);
    }
#endif

} // namespace float
} // namespace xss
#endif
