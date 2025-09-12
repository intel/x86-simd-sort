#ifndef AVX2_EMU_FUNCS
#define AVX2_EMU_FUNCS

#include <array>
#include <utility>

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

constexpr auto avx2_mask_helper_lut64 = [] {
    std::array<std::array<int64_t, 4>, 16> lut {};
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int64_t, 4> entry {};
        for (int j = 0; j < 4; j++) {
            if (((i >> j) & 1) == 1)
                entry[j] = 0xFFFFFFFFFFFFFFFF;
            else
                entry[j] = 0;
        }
        lut[i] = entry;
    }
    return lut;
}();

constexpr auto avx2_mask_helper_lut32_half = [] {
    std::array<std::array<int32_t, 4>, 16> lut {};
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int32_t, 4> entry {};
        for (int j = 0; j < 4; j++) {
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

constexpr auto avx2_compressstore_lut32_half_gen = [] {
    std::array<std::array<std::array<int32_t, 4>, 16>, 2> lutPair {};
    auto &permLut = lutPair[0];
    auto &leftLut = lutPair[1];
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int32_t, 4> indices {};
        std::array<int32_t, 4> leftEntry = {0, 0, 0, 0};
        int right = 3;
        int left = 0;
        for (int j = 0; j < 4; j++) {
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

constexpr auto avx2_compressstore_lut32_half_perm
        = avx2_compressstore_lut32_half_gen[0];
constexpr auto avx2_compressstore_lut32_half_left
        = avx2_compressstore_lut32_half_gen[1];

constexpr auto avx2_compressstore_lut64_gen = [] {
    std::array<std::array<int32_t, 8>, 16> permLut {};
    std::array<std::array<int64_t, 4>, 16> leftLut {};
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int32_t, 8> indices {};
        std::array<int64_t, 4> leftEntry = {0, 0, 0, 0};
        int right = 7;
        int left = 0;
        for (int j = 0; j < 4; j++) {
            bool ge = (i >> j) & 1;
            if (ge) {
                indices[right] = 2 * j + 1;
                indices[right - 1] = 2 * j;
                right -= 2;
            }
            else {
                indices[left + 1] = 2 * j + 1;
                indices[left] = 2 * j;
                leftEntry[left / 2] = 0xFFFFFFFFFFFFFFFF;
                left += 2;
            }
        }
        permLut[i] = indices;
        leftLut[i] = leftEntry;
    }
    return std::make_pair(permLut, leftLut);
}();
constexpr auto avx2_compressstore_lut64_perm
        = avx2_compressstore_lut64_gen.first;
constexpr auto avx2_compressstore_lut64_left
        = avx2_compressstore_lut64_gen.second;

X86_SIMD_SORT_FORCE_INLINE
__m256i convert_int_to_avx2_mask(int32_t m)
{
    return _mm256_loadu_si256(
            (const __m256i *)avx2_mask_helper_lut32[m].data());
}

X86_SIMD_SORT_FORCE_INLINE
int32_t convert_avx2_mask_to_int(__m256i m)
{
    return _mm256_movemask_ps(_mm256_castsi256_ps(m));
}

X86_SIMD_SORT_FORCE_INLINE
__m256i convert_int_to_avx2_mask_64bit(int32_t m)
{
    return _mm256_loadu_si256(
            (const __m256i *)avx2_mask_helper_lut64[m].data());
}

X86_SIMD_SORT_FORCE_INLINE
int32_t convert_avx2_mask_to_int_64bit(__m256i m)
{
    return _mm256_movemask_pd(_mm256_castsi256_pd(m));
}

X86_SIMD_SORT_FORCE_INLINE
__m128i convert_int_to_avx2_mask_half(int32_t m)
{
    return _mm_loadu_si128(
            (const __m128i *)avx2_mask_helper_lut32_half[m].data());
}

X86_SIMD_SORT_FORCE_INLINE
int32_t convert_avx2_mask_to_int_half(__m128i m)
{
    return _mm_movemask_ps(_mm_castsi128_ps(m));
}

// Emulators for intrinsics missing from AVX2 compared to AVX512
template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_max32(typename avx2_vector<T>::reg_t x)
{
    using vtype = avx2_vector<T>;
    using reg_t = typename vtype::reg_t;

    reg_t inter1 = vtype::max(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    reg_t inter2 = vtype::max(
            inter1, vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(inter1));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter2);
    return std::max(arr[0], arr[7]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_max32_half(typename avx2_half_vector<T>::reg_t x)
{
    using vtype = avx2_half_vector<T>;
    using reg_t = typename vtype::reg_t;

    reg_t inter1 = vtype::max(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter1);
    return std::max(arr[0], arr[3]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_min32(typename avx2_vector<T>::reg_t x)
{
    using vtype = avx2_vector<T>;
    using reg_t = typename vtype::reg_t;

    reg_t inter1 = vtype::min(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    reg_t inter2 = vtype::min(
            inter1, vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(inter1));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter2);
    return std::min(arr[0], arr[7]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_min32_half(typename avx2_half_vector<T>::reg_t x)
{
    using vtype = avx2_half_vector<T>;
    using reg_t = typename vtype::reg_t;

    reg_t inter1 = vtype::min(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter1);
    return std::min(arr[0], arr[3]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_max64(typename avx2_vector<T>::reg_t x)
{
    using vtype = avx2_vector<T>;
    typename vtype::reg_t inter1 = vtype::max(
            x, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter1);
    return std::max(arr[0], arr[3]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE T avx2_emu_reduce_min64(typename avx2_vector<T>::reg_t x)
{
    using vtype = avx2_vector<T>;
    typename vtype::reg_t inter1 = vtype::min(
            x, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T arr[vtype::numlanes];
    vtype::storeu(arr, inter1);
    return std::min(arr[0], arr[3]);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE void avx2_emu_mask_compressstoreu32(void *base_addr,
                                    typename avx2_vector<T>::opmask_t k,
                                    typename avx2_vector<T>::reg_t reg)
{
    using vtype = avx2_vector<T>;

    T *leftStore = (T *)base_addr;

    int32_t shortMask = convert_avx2_mask_to_int(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut32_perm[shortMask].data());
    const __m256i &left = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut32_left[shortMask].data());

    typename vtype::reg_t temp = vtype::permutexvar(perm, reg);

    vtype::mask_storeu(leftStore, left, temp);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE void avx2_emu_mask_compressstoreu32_half(
        void *base_addr,
        typename avx2_half_vector<T>::opmask_t k,
        typename avx2_half_vector<T>::reg_t reg)
{
    using vtype = avx2_half_vector<T>;

    T *leftStore = (T *)base_addr;

    int32_t shortMask = convert_avx2_mask_to_int_half(k);
    const __m128i &perm = _mm_loadu_si128(
            (const __m128i *)avx2_compressstore_lut32_half_perm[shortMask]
                    .data());
    const __m128i &left = _mm_loadu_si128(
            (const __m128i *)avx2_compressstore_lut32_half_left[shortMask]
                    .data());

    typename vtype::reg_t temp = vtype::permutexvar(perm, reg);

    vtype::mask_storeu(leftStore, left, temp);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE void avx2_emu_mask_compressstoreu64(void *base_addr,
                                    typename avx2_vector<T>::opmask_t k,
                                    typename avx2_vector<T>::reg_t reg)
{
    using vtype = avx2_vector<T>;

    T *leftStore = (T *)base_addr;

    int32_t shortMask = convert_avx2_mask_to_int_64bit(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_perm[shortMask].data());
    const __m256i &left = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_left[shortMask].data());

    typename vtype::reg_t temp = vtype::cast_from(
            _mm256_permutevar8x32_epi32(vtype::cast_to(reg), perm));

    vtype::mask_storeu(leftStore, left, temp);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE int avx2_double_compressstore32(void *left_addr,
                                void *right_addr,
                                typename avx2_vector<T>::opmask_t k,
                                typename avx2_vector<T>::reg_t reg)
{
    using vtype = avx2_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = convert_avx2_mask_to_int(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut32_perm[shortMask].data());

    typename vtype::reg_t temp = vtype::permutexvar(perm, reg);

    vtype::storeu(leftStore, temp);
    vtype::storeu(rightStore, temp);

    return _mm_popcnt_u32(shortMask);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE int avx2_double_compressstore32_half(void *left_addr,
                                     void *right_addr,
                                     typename avx2_half_vector<T>::opmask_t k,
                                     typename avx2_half_vector<T>::reg_t reg)
{
    using vtype = avx2_half_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = convert_avx2_mask_to_int_half(k);
    const __m128i &perm = _mm_loadu_si128(
            (const __m128i *)avx2_compressstore_lut32_half_perm[shortMask]
                    .data());

    typename vtype::reg_t temp = vtype::permutexvar(perm, reg);

    vtype::storeu(leftStore, temp);
    vtype::storeu(rightStore, temp);

    return _mm_popcnt_u32(shortMask);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE int32_t avx2_double_compressstore64(void *left_addr,
                                    void *right_addr,
                                    typename avx2_vector<T>::opmask_t k,
                                    typename avx2_vector<T>::reg_t reg)
{
    using vtype = avx2_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = convert_avx2_mask_to_int_64bit(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_perm[shortMask].data());

    typename vtype::reg_t temp = vtype::cast_from(
            _mm256_permutevar8x32_epi32(vtype::cast_to(reg), perm));

    vtype::storeu(leftStore, temp);
    vtype::storeu(rightStore, temp);

    return _mm_popcnt_u32(shortMask);
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE typename avx2_vector<T>::reg_t avx2_emu_max(typename avx2_vector<T>::reg_t x,
                                            typename avx2_vector<T>::reg_t y)
{
    using vtype = avx2_vector<T>;
    typename vtype::opmask_t nlt = vtype::gt(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(nlt)));
}

template <typename T>
X86_SIMD_SORT_FORCE_INLINE typename avx2_vector<T>::reg_t avx2_emu_min(typename avx2_vector<T>::reg_t x,
                                            typename avx2_vector<T>::reg_t y)
{
    using vtype = avx2_vector<T>;
    typename vtype::opmask_t nlt = vtype::gt(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(nlt)));
}

#endif
