#ifndef XSS_NETWORK_QSORT
#define XSS_NETWORK_QSORT

namespace xss {

template <typename vtype,
          int64_t numVecs,
          typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void bitonic_clean_n_vec(zmm_t *ymm)
{
#pragma GCC unroll 64
    for (int num = numVecs / 2; num >= 2; num /= 2) {
#pragma GCC unroll 64
        for (int j = 0; j < numVecs; j += num) {
#pragma GCC unroll 64
            for (int i = 0; i < num / 2; i++) {
                COEX<vtype>(ymm[i + j], ymm[i + j + num / 2]);
            }
        }
    }
}

template <typename vtype,
          int64_t numVecs,
          typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_n_vec(zmm_t *ymm)
{
    // Do the reverse part
    if (numVecs == 2) {
        ymm[1] = vtype::reverse(ymm[1]);
        COEX<vtype>(ymm[0], ymm[1]);
    }
    else if (numVecs > 2) {
// Reverse upper half
#pragma GCC unroll 64
        for (int i = 0; i < numVecs / 2; i++) {
            zmm_t rev = vtype::reverse(ymm[numVecs - i - 1]);
            zmm_t maxV = vtype::max(ymm[i], rev);
            zmm_t minV = vtype::min(ymm[i], rev);
            ymm[numVecs - i - 1] = vtype::reverse(maxV);
            ymm[i] = minV;
        }
    }

    // Call cleaner
    bitonic_clean_n_vec<vtype, numVecs>(ymm);

// Now do bitonic_merge_ymm_32bit
#pragma GCC unroll 64
    for (int i = 0; i < numVecs; i++) {
        ymm[i] = vtype::bitonic_merge(ymm[i]);
    }
}

template <typename vtype,
          int64_t numVecs,
          int64_t numPer = 2,
          typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void bitonic_fullmerge_n_vec(zmm_t *ymm)
{
    if (numPer > numVecs)
        return;
    else {
#pragma GCC unroll 64
        for (int i = 0; i < numVecs / numPer; i++) {
            bitonic_merge_n_vec<vtype, numPer>(ymm + i * numPer);
        }
        bitonic_fullmerge_n_vec<vtype, numVecs, (numPer * 2) & 0xFFFFFFFF>(ymm);
    }
}

template <typename vtype, int numVecs, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void sort_n_vec(typename vtype::type_t *arr, int32_t N)
{
    if (numVecs > 1 && N * 2 <= numVecs * vtype::numlanes) {
        sort_n_vec<vtype, numVecs / 2>(arr, N);
        return;
    }

    zmm_t vecs[numVecs];

// Unmasked part of the load
#pragma GCC unroll 64
    for (int i = 0; i < numVecs / 2; i++) {
        vecs[i] = vtype::loadu(arr + i * vtype::numlanes);
    }
// Masked part of the load
#pragma GCC unroll 64
    for (int i = numVecs / 2; i < numVecs; i++) {
        int64_t num_to_write
                = std::min((int64_t)std::max(0, N - i * vtype::numlanes),
                           (int64_t)vtype::numlanes);
        typename vtype::opmask_t load_mask = ((0x1ull << num_to_write) - 0x1ull)
                & ((0x1ull << vtype::numlanes) - 0x1ull);
        vecs[i] = vtype::mask_loadu(
                vtype::zmm_max(), load_mask, arr + i * vtype::numlanes);
    }

// Sort each loaded vector
#pragma GCC unroll 64
    for (int i = 0; i < numVecs; i++) {
        vecs[i] = vtype::sort_vec(vecs[i]);
    }

    // Run the full merger
    bitonic_fullmerge_n_vec<vtype, numVecs>(&vecs[0]);

// Unmasked part of the store
#pragma GCC unroll 64
    for (int i = 0; i < numVecs / 2; i++) {
        vtype::storeu(arr + i * vtype::numlanes, vecs[i]);
    }
// Masked part of the store
#pragma GCC unroll 64
    for (int i = numVecs / 2; i < numVecs; i++) {
        int64_t num_to_write
                = std::min((int64_t)std::max(0, N - i * vtype::numlanes),
                           (int64_t)vtype::numlanes);
        typename vtype::opmask_t load_mask = ((0x1ull << num_to_write) - 0x1ull)
                & ((0x1ull << vtype::numlanes) - 0x1ull);
        vtype::mask_storeu(arr + i * vtype::numlanes, load_mask, vecs[i]);
    }
}

template <typename vtype, int64_t maxN>
X86_SIMD_SORT_INLINE void sort_n(typename vtype::type_t *arr, int N)
{
    sort_n_vec<vtype, maxN / vtype::numlanes>(arr, N);
}
} // namespace xss

#endif