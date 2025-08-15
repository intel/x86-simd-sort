#include "c-api-headers.h"

#ifdef XSS_EXPORTING
#	if defined(__MINGW64__)
#		define XSS_C_EXPORT __declspec(dllexport)
#	else
#		define XSS_C_EXPORT __attribute__((visibility("default")))
#	endif
#	define XSS_C_BODY(body) { try { body; return true; } catch(...) { return false; } }
#else
#	define XSS_C_EXPORT XSS_DLL_IMPORT
#	define XSS_C_BODY(body) ;
#endif

#define XSS_XI1(n,t) XSS_C_EXPORT XSS_QSORT_HEADER_INT(n,t) XSS_C_BODY(x86simdsort::qsort(ar, size, false, descending))
#define XSS_XF1(n,t) XSS_C_EXPORT XSS_QSORT_HEADER_FLT(n,t) XSS_C_BODY(x86simdsort::qsort(ar, size, hasnan, descending))
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_C_EXPORT XSS_QSELECT_HEADER_INT(n,t) XSS_C_BODY(x86simdsort::qselect(ar, k, size, false, descending))
#define XSS_XF1(n,t) XSS_C_EXPORT XSS_QSELECT_HEADER_FLT(n,t) XSS_C_BODY(x86simdsort::qselect(ar, k, size, hasnan, descending))
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_C_EXPORT XSS_QPSORT_HEADER_INT(n,t) XSS_C_BODY(x86simdsort::partial_qsort(ar, k, size, false, descending))
#define XSS_XF1(n,t) XSS_C_EXPORT XSS_QPSORT_HEADER_FLT(n,t) XSS_C_BODY(x86simdsort::partial_qsort(ar, k, size, hasnan, descending))
#include "x-macro1.h"

#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVSORT_HEADER_INT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_qsort(keys, vals, size, false, descending))
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVSORT_HEADER_FLT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_qsort(keys, vals, size, hasnan, descending))
#include "x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVSEL_HEADER_INT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_select(keys, vals, k, size, false, descending))
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVSEL_HEADER_FLT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_select(keys, vals, k, size, hasnan, descending))
#include "x-macro2.h"

#define XSS_XI2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVPSORT_HEADER_INT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_partial_sort(keys, vals, k, size, false, descending))
#define XSS_XF2(n1,t1, n2,t2) XSS_C_EXPORT XSS_QKVPSORT_HEADER_FLT(n1,t1,n2,t2) XSS_C_BODY(x86simdsort::keyvalue_partial_sort(keys, vals, k, size, hasnan, descending))
#include "x-macro2.h"
