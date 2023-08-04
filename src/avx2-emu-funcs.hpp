#ifndef AVX2_EMU_FUNCS
#define AVX2_EMU_FUNCS

#include <array>
#include <utility>
#include "xss-common-qsort.h"

namespace xss {
namespace avx2 {

constexpr auto avx2_mask_helper_lut32 = [] {
    std::array<std::array<int32_t, 8>, 256> lut {};
    for (int64_t i = 0; i <= 0xFF; i++) {
        std::array<int32_t, 8> entry {};
        for (int j = 0; j < 8; j++) {
            if (((i >> j) & 1) == 1)
                entry[j] = 0xFFFFFFFF;
            else
                entry[j] = 0;
        }
        lut[i] = entry;
    }
    return lut;
}();

constexpr auto avx2_compressstore_lut32_gen = [] {
    std::array<std::array<std::array<int32_t, 8>, 256>, 2> lutPair {};
    auto &permLut = lutPair[0];
    auto &leftLut = lutPair[1];
    for (int64_t i = 0; i <= 0xFF; i++) {
        std::array<int32_t, 8> indices {};
        std::array<int32_t, 8> leftEntry = {0, 0, 0, 0, 0, 0, 0, 0};
        int right = 7;
        int left = 0;
        for (int j = 0; j < 8; j++) {
            bool ge = (i >> j) & 1;
            if (ge) {
                indices[right] = j;
                right--;
            }
            else {
                indices[left] = j;
                leftEntry[left] = 0xFFFFFFFF;
                left++;
            }
        }
        permLut[i] = indices;
        leftLut[i] = leftEntry;
    }
    return lutPair;
}();
constexpr auto avx2_compressstore_lut32_perm = avx2_compressstore_lut32_gen[0];
constexpr auto avx2_compressstore_lut32_left = avx2_compressstore_lut32_gen[1];

struct avx2_mask_helper32 {
    __m256i mask;

    avx2_mask_helper32() = default;
    avx2_mask_helper32(int m)
    {
        mask = converter(m);
    }
    avx2_mask_helper32(__m256i m)
    {
        mask = m;
    }
    operator __m256i()
    {
        return mask;
    }
    operator int32_t()
    {
        return converter(mask);
    }
    __m256i operator=(int m)
    {
        mask = converter(m);
        return mask;
    }

private:
    __m256i converter(int m)
    {
        return _mm256_loadu_si256(
                (const __m256i *)avx2_mask_helper_lut32[m].data());
    }

    int32_t converter(__m256i m)
    {
        return _mm256_movemask_ps(_mm256_castsi256_ps(m));
    }
};
static __m256i operator~(const avx2_mask_helper32 x)
{
    return ~x.mask;
}

// Emulators for intrinsics missing from AVX2 compared to AVX512
template <typename T>
T avx2_emu_reduce_max32(typename ymm_vector<T>::reg_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::reg_t inter1 = vtype::max(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    typename vtype::reg_t inter2 = vtype::permutevar(
            inter1, _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0));
    typename vtype::reg_t inter3 = vtype::max(
            inter2, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(inter2));
    T can1 = vtype::template extract<0>(inter3);
    T can2 = vtype::template extract<2>(inter3);
    return std::max(can1, can2);
}

template <typename T>
T avx2_emu_reduce_min32(typename ymm_vector<T>::reg_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::reg_t inter1 = vtype::min(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    typename vtype::reg_t inter2 = vtype::permutevar(
            inter1, _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0));
    typename vtype::reg_t inter3 = vtype::min(
            inter2, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(inter2));
    T can1 = vtype::template extract<0>(inter3);
    T can2 = vtype::template extract<2>(inter3);
    return std::min(can1, can2);
}

template <typename T>
typename ymm_vector<T>::opmask_t
avx2_emu_fpclassify64(typename ymm_vector<T>::reg_t x, int mask)
{
    using vtype = ymm_vector<T>;
    T store[vtype::numlanes];
    vtype::storeu(&store[0], x);
    int64_t res[vtype::numlanes];

    for (int i = 0; i < vtype::numlanes; i++) {
        bool flagged = scalar_emu_fpclassify(store[i]);
        res[i] = 0xFFFFFFFFFFFFFFFF;
    }
    return vtype::loadu(res);
}

template <typename T>
void avx2_emu_mask_compressstoreu(void *base_addr,
                                  typename ymm_vector<T>::opmask_t k,
                                  typename ymm_vector<T>::reg_t reg)
{
    using vtype = ymm_vector<T>;
    T *storage = (T *)base_addr;
    int32_t mask[vtype::numlanes];
    T data[vtype::numlanes];

    _mm256_storeu_si256((__m256i *)&mask[0], k);
    vtype::storeu(&data[0], reg);

#pragma GCC unroll 8
    for (int i = 0; i < vtype::numlanes; i++) {
        if (mask[i]) {
            *storage = data[i];
            storage++;
        }
    }
}

template <typename T>
int32_t avx2_double_compressstore32(void *left_addr,
                                    void *right_addr,
                                    typename ymm_vector<T>::opmask_t k,
                                    typename ymm_vector<T>::reg_t reg)
{
    using vtype = ymm_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = avx2_mask_helper32(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut32_perm[shortMask].data());
    const __m256i &left = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut32_left[shortMask].data());

    typename vtype::reg_t temp = vtype::permutevar(reg, perm);

    vtype::mask_storeu(leftStore, left, temp);
    vtype::mask_storeu(rightStore - vtype::numlanes, ~left, temp);

    return _mm_popcnt_u32(shortMask);
}

template <typename T>
int64_t avx2_emu_popcnt(__m256i reg)
{
    using vtype = ymm_vector<T>;

    int32_t data[vtype::numlanes];
    _mm256_storeu_si256((__m256i *)&data[0], reg);

    int64_t pop = 0;
    for (int i = 0; i < vtype::numlanes; i++) {
        pop += _popcnt32(data[i]);
    }
    return pop;
}

template <typename T>
typename ymm_vector<T>::reg_t avx2_emu_max(typename ymm_vector<T>::reg_t x,
                                           typename ymm_vector<T>::reg_t y)
{
    using vtype = ymm_vector<T>;
    typename vtype::opmask_t nlt = vtype::ge(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(nlt)));
}

template <typename T>
typename ymm_vector<T>::reg_t avx2_emu_min(typename ymm_vector<T>::reg_t x,
                                           typename ymm_vector<T>::reg_t y)
{
    using vtype = ymm_vector<T>;
    typename vtype::opmask_t nlt = vtype::ge(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(nlt)));
}
} // namespace avx2
} // namespace x86_simd_sort

#endif