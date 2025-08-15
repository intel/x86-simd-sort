#pragma once

#define XSS_C_EXP_NAME1(name, ty) c_xss_##name##_##ty
#define XSS_C_EXP_NAME2(name, ty1, ty2) c_xss_##name##_##ty1##_##ty2

#define XSS_QSORT_HEADER_INT(n,t)   bool XSS_C_EXP_NAME1(qsort, n)(t *ar, uint64_t size, bool descending)
#define XSS_QSORT_HEADER_FLT(n,t)   bool XSS_C_EXP_NAME1(qsort, n)(t *ar, uint64_t size, bool hasnan, bool descending)

#define XSS_QSELECT_HEADER_INT(n,t)   bool XSS_C_EXP_NAME1(qselect, n)(t *ar, uint64_t k, uint64_t size, bool descending)
#define XSS_QSELECT_HEADER_FLT(n,t)   bool XSS_C_EXP_NAME1(qselect, n)(t *ar, uint64_t k, uint64_t size, bool hasnan, bool descending)

#define XSS_QPSORT_HEADER_INT(n,t)   bool XSS_C_EXP_NAME1(partial_qsort, n)(t *ar, uint64_t k, uint64_t size, bool descending)
#define XSS_QPSORT_HEADER_FLT(n,t)   bool XSS_C_EXP_NAME1(partial_qsort, n)(t *ar, uint64_t k, uint64_t size, bool hasnan, bool descending)

#define XSS_QKVSORT_HEADER_INT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_qsort, n1, n2)(t1 *keys, t2* vals, uint64_t size, bool descending)
#define XSS_QKVSORT_HEADER_FLT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_qsort, n1, n2)(t1 *keys, t2* vals, uint64_t size, bool hasnan, bool descending)

#define XSS_QKVSEL_HEADER_INT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_qselect, n1, n2)(t1 *keys, t2* vals, uint64_t k, uint64_t size, bool descending)
#define XSS_QKVSEL_HEADER_FLT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_qselect, n1, n2)(t1 *keys, t2* vals, uint64_t k, uint64_t size, bool hasnan, bool descending)

#define XSS_QKVPSORT_HEADER_INT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_partial_qsort, n1, n2)(t1 *keys, t2* vals, uint64_t k, uint64_t size, bool descending)
#define XSS_QKVPSORT_HEADER_FLT(n1,t1,n2,t2)   bool XSS_C_EXP_NAME2(keyvalue_partial_qsort, n1, n2)(t1 *keys, t2* vals, uint64_t k, uint64_t size, bool hasnan, bool descending)



