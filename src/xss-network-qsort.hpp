#ifndef XSS_NETWORK_QSORT
#define XSS_NETWORK_QSORT

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void bitonic_clean_n_vec(reg_t *regs)
{
#pragma GCC unroll 64
    for (int num = numVecs / 2; num >= 2; num /= 2) {
#pragma GCC unroll 64
        for (int j = 0; j < numVecs; j += num) {
#pragma GCC unroll 64
            for (int i = 0; i < num / 2; i++) {
                COEX<vtype>(regs[i + j], regs[i + j + num / 2]);
            }
        }
    }
}

template <typename vtype,
          int64_t numVecs,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void bitonic_merge_n_vec(reg_t *regs)
{
    // Do the reverse part
    if constexpr (numVecs == 2) {
        regs[1] = vtype::reverse(regs[1]);
        COEX<vtype>(regs[0], regs[1]);
    }
    else if constexpr (numVecs > 2) {
// Reverse upper half
#pragma GCC unroll 64
        for (int i = 0; i < numVecs / 2; i++) {
            reg_t rev = vtype::reverse(regs[numVecs - i - 1]);
            reg_t maxV = vtype::max(regs[i], rev);
            reg_t minV = vtype::min(regs[i], rev);
            regs[numVecs - i - 1] = vtype::reverse(maxV);
            regs[i] = minV;
        }
    }

    // Call cleaner
    bitonic_clean_n_vec<vtype, numVecs>(regs);

// Now do bitonic_merge
#pragma GCC unroll 64
    for (int i = 0; i < numVecs; i++) {
        regs[i] = vtype::bitonic_merge(regs[i]);
    }
}

template <typename vtype,
          int64_t numVecs,
          int64_t numPer = 2,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void bitonic_fullmerge_n_vec(reg_t *regs)
{
    if constexpr (numPer > numVecs)
        return;
    else {
#pragma GCC unroll 64
        for (int i = 0; i < numVecs / numPer; i++) {
            bitonic_merge_n_vec<vtype, numPer>(regs + i * numPer);
        }
        bitonic_fullmerge_n_vec<vtype, numVecs, numPer * 2>(regs);
    }
}

template <typename vtype, int numVecs, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE void sort_n_vec(typename vtype::type_t *arr, int32_t N)
{
    if (numVecs > 1 && N * 2 <= numVecs * vtype::numlanes) {
        sort_n_vec<vtype, numVecs / 2>(arr, N);
        return;
    }

    reg_t vecs[numVecs];
    
    // Generate masks for loading and storing
    typename vtype::opmask_t ioMasks[numVecs / 2];
    #pragma GCC unroll 64
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        int64_t num_to_read
                = std::min((int64_t)std::max(0, N - i * vtype::numlanes),
                           (int64_t)vtype::numlanes);
        ioMasks[j] = ((0x1ull << num_to_read) - 0x1ull);
    }

// Unmasked part of the load
#pragma GCC unroll 64
    for (int i = 0; i < numVecs / 2; i++) {
        vecs[i] = vtype::loadu(arr + i * vtype::numlanes);
    }
// Masked part of the load
#pragma GCC unroll 64
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        vecs[i] = vtype::mask_loadu(
                vtype::zmm_max(), ioMasks[j], arr + i * vtype::numlanes);
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
    for (int i = numVecs / 2, j = 0; i < numVecs; i++, j++) {
        vtype::mask_storeu(arr + i * vtype::numlanes, ioMasks[j], vecs[i]);
    }
}

template <typename vtype, int64_t maxN>
X86_SIMD_SORT_INLINE void sort_n(typename vtype::type_t *arr, int N)
{
    constexpr int numVecs = maxN / vtype::numlanes;
    constexpr bool isMultiple = (maxN == (vtype::numlanes * numVecs));
    constexpr bool powerOfTwo = (numVecs != 0 && !(numVecs & (numVecs - 1)));
    static_assert(powerOfTwo == true && isMultiple == true,
                  "maxN must be vtype::numlanes times a power of 2");

    sort_n_vec<vtype, numVecs>(arr, N);
}

#endif