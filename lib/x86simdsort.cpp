#include "x86simdsort.h"
#include "x86simdsort-internal.h"
#include "x86simdsort-scalar.h"
#include <algorithm>
#include <iostream>
#include <string>

static int check_cpu_feature_support(std::string_view cpufeature)
{
    const char* disable_avx512 = std::getenv("XSS_DISABLE_AVX512");

    if ((cpufeature == "avx512_spr") && (!disable_avx512))
#ifdef __FLT16_MAX__
        return __builtin_cpu_supports("avx512f")
                && __builtin_cpu_supports("avx512fp16")
                && __builtin_cpu_supports("avx512vbmi2");
#else
        return 0;
#endif
    else if ((cpufeature == "avx512_icl") && (!disable_avx512))
        return __builtin_cpu_supports("avx512f")
                && __builtin_cpu_supports("avx512vbmi2")
                && __builtin_cpu_supports("avx512bw")
                && __builtin_cpu_supports("avx512vl");
    else if ((cpufeature == "avx512_skx") && (!disable_avx512))
        return __builtin_cpu_supports("avx512f")
                && __builtin_cpu_supports("avx512dq")
                && __builtin_cpu_supports("avx512vl");
    else if (cpufeature == "avx2")
        return __builtin_cpu_supports("avx2");

    return 0;
}

std::string_view static find_preferred_cpu(
        std::initializer_list<std::string_view> cpulist)
{
    for (auto cpu : cpulist) {
        if (check_cpu_feature_support(cpu)) return cpu;
    }
    return "scalar";
}

constexpr bool
dispatch_requested(std::string_view cpurequested,
                   std::initializer_list<std::string_view> cpulist)
{
    for (auto cpu : cpulist) {
        if (cpu.find(cpurequested) != std::string_view::npos) return true;
    }
    return false;
}

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

#define DECLARE_INTERNAL_qsort(TYPE) \
    static void (*internal_qsort##TYPE)(TYPE *, size_t) = NULL; \
    template <> \
    void qsort(TYPE *arr, size_t arrsize) \
    { \
        (*internal_qsort##TYPE)(arr, arrsize); \
    }

#define DECLARE_INTERNAL_qselect(TYPE) \
    static void (*internal_qselect##TYPE)(TYPE *, size_t, size_t, bool) \
            = NULL; \
    template <> \
    void qselect(TYPE *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        (*internal_qselect##TYPE)(arr, k, arrsize, hasnan); \
    }

#define DECLARE_INTERNAL_partial_qsort(TYPE) \
    static void (*internal_partial_qsort##TYPE)(TYPE *, size_t, size_t, bool) \
            = NULL; \
    template <> \
    void partial_qsort(TYPE *arr, size_t k, size_t arrsize, bool hasnan) \
    { \
        (*internal_partial_qsort##TYPE)(arr, k, arrsize, hasnan); \
    }

#define DECLARE_INTERNAL_argsort(TYPE) \
    static std::vector<size_t> (*internal_argsort##TYPE)(TYPE *, size_t) \
            = NULL; \
    template <> \
    std::vector<size_t> argsort(TYPE *arr, size_t arrsize) \
    { \
        return (*internal_argsort##TYPE)(arr, arrsize); \
    }

#define DECLARE_INTERNAL_argselect(TYPE) \
    static std::vector<size_t> (*internal_argselect##TYPE)( \
            TYPE *, size_t, size_t) \
            = NULL; \
    template <> \
    std::vector<size_t> argselect(TYPE *arr, size_t k, size_t arrsize) \
    { \
        return (*internal_argselect##TYPE)(arr, k, arrsize); \
    }

/* runtime dispatch mechanism */
#define DISPATCH(func, TYPE, ...) \
    DECLARE_INTERNAL_##func(TYPE) static __attribute__((constructor)) void \
    CAT(CAT(resolve_, func), TYPE)(void) \
    { \
        CAT(CAT(internal_, func), TYPE) = &xss::scalar::func<TYPE>; \
        __builtin_cpu_init(); \
        std::string_view preferred_cpu = find_preferred_cpu({__VA_ARGS__}); \
        if constexpr (dispatch_requested("avx512", {__VA_ARGS__})) { \
            if (preferred_cpu.find("avx512") != std::string_view::npos) { \
                CAT(CAT(internal_, func), TYPE) = &xss::avx512::func<TYPE>; \
                return; \
            } \
        } \
        else if constexpr (dispatch_requested("avx2", {__VA_ARGS__})) { \
            if (preferred_cpu.find("avx2") != std::string_view::npos) { \
                CAT(CAT(internal_, func), TYPE) = &xss::avx2::func<TYPE>; \
                return; \
            } \
        } \
    }

namespace x86simdsort {
#ifdef __FLT16_MAX__
DISPATCH(qsort, _Float16, "avx512_spr")
DISPATCH(qselect, _Float16, "avx512_spr")
DISPATCH(partial_qsort, _Float16, "avx512_spr")
DISPATCH(argsort, _Float16, "none")
DISPATCH(argselect, _Float16, "none")
#endif

#define DISPATCH_ALL(func, ISA_16BIT, ISA_32BIT, ISA_64BIT) \
    DISPATCH(func, uint16_t, ISA_16BIT) \
    DISPATCH(func, int16_t, ISA_16BIT) \
    DISPATCH(func, float, ISA_32BIT) \
    DISPATCH(func, int32_t, ISA_32BIT) \
    DISPATCH(func, uint32_t, ISA_32BIT) \
    DISPATCH(func, int64_t, ISA_64BIT) \
    DISPATCH(func, uint64_t, ISA_64BIT) \
    DISPATCH(func, double, ISA_64BIT)

DISPATCH_ALL(qsort, ("avx512_icl"), ("avx512_skx"), ("avx512_skx"))
DISPATCH_ALL(qselect, ("avx512_icl"), ("avx512_skx"), ("avx512_skx"))
DISPATCH_ALL(partial_qsort, ("avx512_icl"), ("avx512_skx"), ("avx512_skx"))
DISPATCH_ALL(argsort, "none", "avx512_skx", "avx512_skx")
DISPATCH_ALL(argselect, "none", "avx512_skx", "avx512_skx")

} // namespace x86simdsort
