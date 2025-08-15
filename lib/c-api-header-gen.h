COMMENT This is an auto-generated file
COMMENT This header is intended to be used for shared libraries (so or dll) compiled to the shared_c_api target

PRAGMA ifndef __X86_SIMD_SORT_C_API_H__
PRAGMA define __X86_SIMD_SORT_C_API_H__

EMPTYLINE
PRAGMA include <cstdint>
EMPTYLINE

PRAGMA define XSS_DLL_IMPORT

#include "c-api-headers.h"

EMPTYLINE
COMMENT DLL import declarations

PRAGMA ifdef __cplusplus
extern "C" {
PRAGMA endif
#include "c-api-export.h"
PRAGMA ifdef __cplusplus
} COMMENT extern "C"
PRAGMA endif

EMPTYLINE
PRAGMA ifdef __cplusplus
COMMENT C++ overloaded dispatchers
namespace xss
{

#define XSS_XI1(n,t) inline bool qsort(t* ar, uint64_t sz, bool desc) { return XSS_C_EXP_NAME1(qsort,n)( ar, sz, desc ); }
#define XSS_XF1(n,t) inline bool qsort(t* ar, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME1(qsort,n)( ar, sz, hasnan, desc ); }
#include "c-api-x-macro1.h"

#define XSS_XI1(n,t) inline bool qselect(t* ar, uint64_t k, uint64_t sz, bool desc) { return XSS_C_EXP_NAME1(qselect,n)( ar, k, sz, desc ); }
#define XSS_XF1(n,t) inline bool qselect(t* ar, uint64_t k, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME1(qselect,n)( ar, k, sz, hasnan, desc ); }
#include "c-api-x-macro1.h"

#define XSS_XI1(n,t) inline bool partial_qsort(t* ar, uint64_t k, uint64_t sz, bool desc) { return XSS_C_EXP_NAME1(partial_qsort,n)( ar, k, sz, desc ); }
#define XSS_XF1(n,t) inline bool partial_qsort(t* ar, uint64_t k, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME1(partial_qsort,n)( ar, k, sz, hasnan, desc ); }
#include "c-api-x-macro1.h"

#define XSS_XI2(n1,t1, n2,t2) inline bool keyvalue_qsort(t1* keys, t2* vals, uint64_t sz, bool desc) { return XSS_C_EXP_NAME2(keyvalue_qsort,n1, n2)( keys, vals, sz, desc ); }
#define XSS_XF2(n1,t1, n2,t2) inline bool keyvalue_qsort(t1* keys, t2* vals, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME2(keyvalue_qsort,n1, n2)( keys, vals, sz, hasnan, desc ); }
#include "c-api-x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) inline bool keyvalue_qselect(t1* keys, t2* vals, uint64_t k, uint64_t sz, bool desc) { return XSS_C_EXP_NAME2(keyvalue_qselect,n1, n2)( keys, vals, k, sz, desc ); }
#define XSS_XF2(n1,t1, n2,t2) inline bool keyvalue_qselect(t1* keys, t2* vals, uint64_t k, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME2(keyvalue_qselect,n1, n2)( keys, vals, k, sz, hasnan, desc ); }
#include "c-api-x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) inline bool keyvalue_partial_qsort(t1* keys, t2* vals, uint64_t k, uint64_t sz, bool desc) { return XSS_C_EXP_NAME2(keyvalue_partial_qsort,n1, n2)( keys, vals, k, sz, desc ); }
#define XSS_XF2(n1,t1, n2,t2) inline bool keyvalue_partial_qsort(t1* keys, t2* vals, uint64_t k, uint64_t sz, bool hasnan, bool desc)  { return XSS_C_EXP_NAME2(keyvalue_partial_qsort,n1, n2)( keys, vals, k, sz, hasnan, desc ); }
#include "c-api-x-macro2.h"

} COMMENT namespace xss
 
PRAGMA endif COMMENT ifdef __cplusplus
EMPTYLINE
PRAGMA endif COMMENT ifndef __X86_SIMD_SORT_C_API_H__
