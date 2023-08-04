template <typename vtype, typename mm_t>
X86_SIMD_SORT_INLINE void COEX(mm_t &a, mm_t &b);

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_avx512_16bit(type_t *arr,
                                            const arrsize_t left,
                                            const arrsize_t right)
{
    // median of 32
    arrsize_t size = (right - left) / 32;
    type_t vec_arr[32] = {arr[left],
                          arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size],
                          arr[left + 17 * size],
                          arr[left + 18 * size],
                          arr[left + 19 * size],
                          arr[left + 20 * size],
                          arr[left + 21 * size],
                          arr[left + 22 * size],
                          arr[left + 23 * size],
                          arr[left + 24 * size],
                          arr[left + 25 * size],
                          arr[left + 26 * size],
                          arr[left + 27 * size],
                          arr[left + 28 * size],
                          arr[left + 29 * size],
                          arr[left + 30 * size],
                          arr[left + 31 * size]};
    typename vtype::reg_t rand_vec = vtype::loadu(vec_arr);
    typename vtype::reg_t sort = vtype::sort_vec(rand_vec);
    return ((type_t *)&sort)[16];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_avx512_32bit(type_t *arr,
                                            const arrsize_t left,
                                            const arrsize_t right)
{
    // median of 16
    arrsize_t size = (right - left) / 16;
    using reg_t = typename vtype::reg_t;
    type_t vec_arr[16] = {arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size]};
    reg_t rand_vec = vtype::loadu(vec_arr);
    reg_t sort = vtype::sort_vec(rand_vec);
    // pivot will never be a nan, since there are no nan's!
    return ((type_t *)&sort)[8];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_avx512_64bit(type_t *arr,
                                            const arrsize_t left,
                                            const arrsize_t right)
{
    // median of 8
    arrsize_t size = (right - left) / 8;
    using reg_t = typename vtype::reg_t;
    reg_t rand_vec = vtype::set(arr[left + size],
                                arr[left + 2 * size],
                                arr[left + 3 * size],
                                arr[left + 4 * size],
                                arr[left + 5 * size],
                                arr[left + 6 * size],
                                arr[left + 7 * size],
                                arr[left + 8 * size]);
    // pivot will never be a nan, since there are no nan's!
    reg_t sort = vtype::sort_vec(rand_vec);
    return ((type_t *)&sort)[4];
}

template <typename vtype, int maxN>
X86_SIMD_SORT_INLINE void sort_n(typename vtype::type_t *arr, int N);

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_scalar(type_t *arr,
                                      const arrsize_t left,
                                      const arrsize_t right)
{
    type_t samples[vtype::numlanes];
    
    arrsize_t delta = (right - left) / vtype::numlanes;
    
    for (int i = 0; i < vtype::numlanes; i++){
        samples[i] = arr[left + i * delta];
    }
    
    sort_n<vtype, vtype::numlanes>(samples, vtype::numlanes);
    
    return samples[vtype::numlanes / 2];
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot(type_t *arr,
                                      const arrsize_t left,
                                      const arrsize_t right)
{
    using reg_t = typename vtype::reg_t;
    if constexpr (sizeof(reg_t) == 64){ // AVX512
        if constexpr (vtype::numlanes == 8)
            return get_pivot_avx512_64bit<vtype>(arr, left, right);
        else if constexpr (vtype::numlanes == 16)
            return get_pivot_avx512_32bit<vtype>(arr, left, right);
        else if constexpr (vtype::numlanes == 32)
            return get_pivot_avx512_16bit<vtype>(arr, left, right);
        else
            static_assert(vtype::numlanes == -1, "should not reach here");
    }else if constexpr (sizeof(reg_t) == 32) { // AVX2
        if constexpr (vtype::numlanes == 8){
            return get_pivot_scalar<vtype>(arr, left, right);
        }
    }else{
        static_assert(sizeof(reg_t) == -1, "should not reach here");
    }
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_blocks(type_t *arr,
                                             arrsize_t left,
                                             arrsize_t right)
{

    if (right - left <= 1024) { return get_pivot<vtype>(arr, left, right); }

    using reg_t = typename vtype::reg_t;
    constexpr int numVecs = 5;

    arrsize_t width = (right - vtype::numlanes) - left;
    arrsize_t delta = width / numVecs;

    reg_t vecs[numVecs];
    // Load data
    for (int i = 0; i < numVecs; i++) {
        vecs[i] = vtype::loadu(arr + left + delta * i);
    }

    // Implement sorting network (from https://bertdobbelaere.github.io/sorting_networks.html)
    COEX<vtype>(vecs[0], vecs[3]);
    COEX<vtype>(vecs[1], vecs[4]);

    COEX<vtype>(vecs[0], vecs[2]);
    COEX<vtype>(vecs[1], vecs[3]);

    COEX<vtype>(vecs[0], vecs[1]);
    COEX<vtype>(vecs[2], vecs[4]);

    COEX<vtype>(vecs[1], vecs[2]);
    COEX<vtype>(vecs[3], vecs[4]);

    COEX<vtype>(vecs[2], vecs[3]);

    // Calculate median of the middle vector
    reg_t &vec = vecs[numVecs / 2];
    vec = vtype::sort_vec(vec);

    type_t data[vtype::numlanes];
    vtype::storeu(data, vec);
    return data[vtype::numlanes / 2];
}
