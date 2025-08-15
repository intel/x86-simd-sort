#ifndef XSS_XI2
#	error XSS_XI3 must be defined
#endif

#ifndef XSS_XF2
#	error XSS_XF2 must be defined
#endif

#define XSS_XI1(n,t) XSS_XI2(int32, int32_t, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_XI2(uint32, uint32_t, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_XI2(int64, int64_t, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_XI2(uint64, uint64_t, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_XF2(float, float, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#define XSS_XI1(n,t) XSS_XF2(double, double, n, t)
#define XSS_XF1(n,t) XSS_XI1(n, t)
#include "x-macro1.h"

#undef XSS_XI2
#undef XSS_XF2



