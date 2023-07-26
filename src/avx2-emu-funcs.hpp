#ifndef AVX2_EMU_FUNCS
#define AVX2_EMU_FUNCS

namespace x86_simd_sort {
namespace avx2 {

class avx2_mask_helper_lut_gen32 {
public:
    __m256i lut[256];

private:
    void build_lut()
    {
        for (int64_t i = 0; i <= 0xFF; i++) {
            int32_t entry[8];
            for (int j = 0; j < 8; j++) {
                if (((i >> j) & 1) == 1)
                    entry[j] = 0xFFFFFFFF;
                else
                    entry[j] = 0;
            }
            lut[i] = _mm256_loadu_si256((__m256i *)&entry[0]);
        }
    }

public:
    avx2_mask_helper_lut_gen32()
    {
        build_lut();
    }
};
avx2_mask_helper_lut_gen32 avx2_mask_helper_lut32
        = avx2_mask_helper_lut_gen32();

struct avx2_mask_helper32 {
    __m256i mask;

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
        return avx2_mask_helper_lut32.lut[m];
    }

    int32_t converter(__m256i m)
    {
        return _mm256_movemask_ps(_mm256_castsi256_ps(m));
    }
};
__m256i operator~(const avx2_mask_helper32 x)
{
    return ~x.mask;
}

class avx2_mask_helper_lut_gen64 {
public:
    __m256i lut[16];

private:
    void build_lut()
    {
        for (int64_t i = 0; i <= 0xF; i++) {
            int64_t entry[4];
            for (int j = 0; j < 4; j++) {
                if (((i >> j) & 1) == 1)
                    entry[j] = 0xFFFFFFFFFFFFFFFF;
                else
                    entry[j] = 0;
            }
            lut[i] = _mm256_loadu_si256((__m256i *)&entry[0]);
        }
    }

public:
    avx2_mask_helper_lut_gen64()
    {
        build_lut();
    }
};
avx2_mask_helper_lut_gen64 avx2_mask_helper_lut64
        = avx2_mask_helper_lut_gen64();

struct avx2_mask_helper64 {
    __m256i mask;

    avx2_mask_helper64(int m)
    {
        mask = converter(m);
    }
    avx2_mask_helper64(__m256i m)
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
        return avx2_mask_helper_lut64.lut[m];
    }

    int32_t converter(__m256i m)
    {
        return _mm256_movemask_pd(_mm256_castsi256_pd(m));
    }
};
__m256i operator~(const avx2_mask_helper64 x)
{
    return ~x.mask;
}

// Emulators for intrinsics missing from AVX2 compared to AVX512
template <typename T>
T avx2_emu_reduce_max32(typename ymm_vector<T>::ymm_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::ymm_t inter1 = vtype::max(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    typename vtype::ymm_t inter2 = vtype::permutevar(
            inter1, _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0));
    typename vtype::ymm_t inter3 = vtype::max(
            inter2, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(inter2));
    T can1 = vtype::template extract<0>(inter3);
    T can2 = vtype::template extract<2>(inter3);
    return std::max(can1, can2);
}

template <typename T>
T avx2_emu_reduce_max64(typename ymm_vector<T>::ymm_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::ymm_t inter1 = vtype::max(
            x, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T can1 = vtype::template extract<0>(inter1);
    T can2 = vtype::template extract<2>(inter1);
    return std::max<T>(can1, can2);
}

template <typename T>
T avx2_emu_reduce_min32(typename ymm_vector<T>::ymm_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::ymm_t inter1 = vtype::min(
            x, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    typename vtype::ymm_t inter2 = vtype::permutevar(
            inter1, _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0));
    typename vtype::ymm_t inter3 = vtype::min(
            inter2, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(inter2));
    T can1 = vtype::template extract<0>(inter3);
    T can2 = vtype::template extract<2>(inter3);
    return std::min(can1, can2);
}

template <typename T>
T avx2_emu_reduce_min64(typename ymm_vector<T>::ymm_t x)
{
    using vtype = ymm_vector<T>;
    typename vtype::ymm_t inter1 = vtype::min(
            x, vtype::template permutexvar<SHUFFLE_MASK(2, 3, 0, 1)>(x));
    T can1 = vtype::template extract<0>(inter1);
    T can2 = vtype::template extract<2>(inter1);
    return std::min<T>(can1, can2);
}

// TODO this is not perfect! does not distinguish types of NaNs, types of zeros, types of infinities
template <typename T>
bool scalar_emu_fpclassify(T x, int mask)
{
    int classification = std::fpclassify(x);
    int unmasked = 0;

    switch (classification) {
        case FP_INFINITE: unmasked |= 0x08 | 0x10; break;
        case FP_ZERO: unmasked |= 0x02 | 0x04; break;
        case FP_NAN: unmasked |= 0x01 | 0x80; break;
        case FP_SUBNORMAL: unmasked |= 0x20; break;
    }
    if (x < 0) unmasked |= 0x40;

    return (unmasked & mask) != 0;
}

template <typename T>
typename ymm_vector<T>::opmask_t
avx2_emu_fpclassify32(typename ymm_vector<T>::ymm_t x, int mask)
{
    using vtype = ymm_vector<T>;
    T store[vtype::numlanes];
    vtype::storeu(&store[0], x);
    int32_t res[vtype::numlanes];

    for (int i = 0; i < vtype::numlanes; i++) {
        bool flagged = scalar_emu_fpclassify(store[i]);
        res[i] = 0xFFFFFFFF;
    }
    return vtype::loadu(res);
}

template <typename T>
typename ymm_vector<T>::opmask_t
avx2_emu_fpclassify64(typename ymm_vector<T>::ymm_t x, int mask)
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

__m256i avx2_emu_permute_64bit(__m256i x, __m256i mask)
{
    return _mm256_permutevar8x32_epi32(x, mask);
}

__m256d avx2_emu_permute_64bit(__m256d x, __m256i mask)
{
    return _mm256_castsi256_pd(
            _mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), mask));
}

class avx2_compressstore_lut32_gen {
public:
    __m256i permLut[256];
    __m256i leftLut[256];

private:
    void build_lut()
    {
        for (int64_t i = 0; i <= 0xFF; i++) {
            int32_t indices[8];
            int32_t leftEntry[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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
            permLut[i] = _mm256_loadu_si256((__m256i *)&indices[0]);
            leftLut[i] = _mm256_loadu_si256((__m256i *)&leftEntry[0]);
        }
    }

public:
    avx2_compressstore_lut32_gen()
    {
        build_lut();
    }
};

class avx2_compressstore_lut64_gen {
public:
    __m256i permLut[16];
    __m256i leftLut[16];

private:
    void build_lut()
    {
        for (int64_t i = 0; i <= 0xF; i++) {
            int32_t indices[8];
            int64_t leftEntry[4] = {0, 0, 0, 0};
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
            permLut[i] = _mm256_loadu_si256((__m256i *)&indices[0]);
            leftLut[i] = _mm256_loadu_si256((__m256i *)&leftEntry[0]);
        }
    }

public:
    avx2_compressstore_lut64_gen()
    {
        build_lut();
    }
};

const avx2_compressstore_lut32_gen avx2_compressstore_lut32
        = avx2_compressstore_lut32_gen();
const avx2_compressstore_lut64_gen avx2_compressstore_lut64
        = avx2_compressstore_lut64_gen();

template <typename T>
void avx2_emu_mask_compressstoreu(void *base_addr,
                                  typename ymm_vector<T>::opmask_t k,
                                  typename ymm_vector<T>::ymm_t reg)
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

/*template <typename T>
void avx2_emu_mask_compressstoreu(void * base_addr, typename ymm_vector<T>::opmask_t k, typename ymm_vector<T>::ymm_t reg){
	using vtype = ymm_vector<T>;
    T * storage = (T *) base_addr;
    int32_t mask[vtype::numlanes];
	T data[vtype::numlanes];

    _mm256_storeu_si256((__m256i *)&mask[0], k);
	vtype::storeu(&data[0], reg);

	#pragma GCC unroll 8
	for (int i = 0; i < vtype::numlanes; i++){
		if (mask[i]){
			*storage = data[i];
			storage++;
		}
	}
}*/

template <typename T>
int32_t avx2_double_compressstore32(void *left_addr,
                                    void *right_addr,
                                    typename ymm_vector<T>::opmask_t k,
                                    typename ymm_vector<T>::ymm_t reg)
{
    using vtype = ymm_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = avx2_mask_helper32(k);
    const __m256i &perm = avx2_compressstore_lut32.permLut[shortMask];
    const __m256i &left = avx2_compressstore_lut32.leftLut[shortMask];

    typename vtype::ymm_t temp = vtype::permutevar(reg, perm);

    vtype::mask_storeu(leftStore, left, temp);
    vtype::mask_storeu(rightStore - vtype::numlanes, ~left, temp);

    return _mm_popcnt_u32(shortMask);
}

template <typename T>
int32_t avx2_double_compressstore64(void *left_addr,
                                    void *right_addr,
                                    typename ymm_vector<T>::opmask_t k,
                                    typename ymm_vector<T>::ymm_t reg)
{
    using vtype = ymm_vector<T>;

    T *leftStore = (T *)left_addr;
    T *rightStore = (T *)right_addr;

    int32_t shortMask = avx2_mask_helper64(k);
    const __m256i &perm = avx2_compressstore_lut64.permLut[shortMask];
    const __m256i &left = avx2_compressstore_lut64.leftLut[shortMask];

    typename vtype::ymm_t temp = avx2_emu_permute_64bit(reg, perm);

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
typename ymm_vector<T>::ymm_t avx2_emu_max(typename ymm_vector<T>::ymm_t x,
                                           typename ymm_vector<T>::ymm_t y)
{
    using vtype = ymm_vector<T>;
    typename vtype::opmask_t nlt = vtype::ge(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(nlt)));
}

template <typename T>
typename ymm_vector<T>::ymm_t avx2_emu_min(typename ymm_vector<T>::ymm_t x,
                                           typename ymm_vector<T>::ymm_t y)
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